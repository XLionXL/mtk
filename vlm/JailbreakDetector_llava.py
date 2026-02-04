from IsolationForest import PyTorchIsolationForest
import os
import torch
from tqdm import tqdm

class JailbreakDetector:
    """
    Jailbreak prompt detector based on K-NB Rank + IsolationForest
    Core logic: Calculate similarity ranking with benign samples, use IsolationForest for anomaly detection
    """

    def __init__(self, model, processor, background_layered_activations, all_labels, flag, n_estimators=100, random_state=42, k_nb=5):
        self.model = model  
        self.processor = processor  
        self.device = model.device  
        self.flag = flag
        self.k_nb = k_nb  
        self.background_activations_by_layer = background_layered_activations
        self.background_labels = all_labels  
        self.num_layers = len(self.background_activations_by_layer) 
        if os.path.exists(f"/HARD-DATA/ZHT/mllm_mtk/{flag}/training_sequences.pt"):
            training_sequences = torch.load(f"./experimental_results/{flag}/training_sequences.pt")
        else:
            training_sequences = self._get_training_sequences()
        y_train = self.background_labels 
        benign_indices = torch.where(y_train == 0)[0]
        benign_training_sequences = training_sequences[benign_indices]
        self.mean = benign_training_sequences.mean(dim=0, keepdim=True)
        self.std = benign_training_sequences.std(dim=0, keepdim=True) + 1e-8  
        X_train = (benign_training_sequences - self.mean) / self.std  
        self.if_model = PyTorchIsolationForest(n_estimators=100, max_samples=256, random_state=42)
        self.if_model.fit(X_train)  

    def predict(self, sentence):
        """
        Predict if a prompt is jailbreak
        Args:
            sentence: Text prompt (e.g., user input question)
        Returns:
            numeric label (0/1), anomaly score
        """
        messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": sentence[1]},
                        {"type": "text", "text": sentence[0]}
                    ]
                },
            ]
        with torch.no_grad(): 
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[1:]
            activations = [layer_hidden_state[0, -1, :].clone() for layer_hidden_state in hidden_states]
        new_activations = torch.stack(activations, dim=0).squeeze(0)
        # Calculate K-NB rank sequence for test sample (layer-wise iteration)
        ranks = self._calculate_single_rank_k_nb(
            new_activations,  # Test sample features of current layer
            self.background_activations_by_layer,  # All background sample features of current layer
            0,  
            self.background_labels,  # Background sample labels
            k=self.k_nb,  # K value (top K similar benign samples)
            device=self.device
        )
        # Feature standardization (consistent with training)
        scaled_sequence = (ranks - self.mean) / self.std  
        # Calculate anomaly score (PyTorch IsolationForest)
        anomaly_score = self.if_model.decision_function(scaled_sequence)[0].item()
        if anomaly_score > 0: 
            pred_label = 1  
        else: 
            pred_label = 0  

        return pred_label, anomaly_score

    def _restructure_activations(self, activations_list):
        """
        Restructure tensor list from sample-layer to layer-sample
        Args:
            activations_list: List of per-sample layer features (GPU tensor list)
                              e.g., [sample1_feats, sample2_feats, ...]
                              sample1_feats = [layer1_tensor, layer2_tensor, ...] (shape: [feat_dim])
        Returns:
            activations_by_layer: List of per-layer sample features (GPU tensor)
                                  e.g., [layer1_tensor, layer2_tensor, ...] (shape: [num_samples, feat_dim])
        """
        if not activations_list:
            return []
        
        # Get total layers (from first sample's feature list length)
        num_layers = len(activations_list[0])
        # Initialize empty list for layer-wise storage
        activations_by_layer = [[] for _ in range(num_layers)]
        
        # Traverse each sample's features and add to corresponding layer list
        for sample_activations in activations_list:
            for i in range(num_layers):
                activations_by_layer[i].append(sample_activations[i])
        
        # Concatenate each layer list to 2D GPU tensor (replace np.array conversion)
        return [torch.stack(layer_acts, dim=0) for layer_acts in activations_by_layer]
    
    def _get_training_sequences(self):
        """Calculate K-NB rank sequence as features for each sample in mixed background pool """
        print(f"? Calculating training sequences for mixed background pool (K-NB Rank, k={self.k_nb})...")
        num_samples = len(self.background_labels)
        num_layers = 32
        
        # Check GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using computing device: {device}")
        
        # Pre-create Tensor to store all sequences (avoid list append then conversion)
        all_sequences = torch.empty((num_samples, num_layers), device=device)
        
        # Keep background activations and labels on GPU
        background_activations_gpu = self.background_activations_by_layer
        background_labels_gpu = self.background_labels
        
        for i in tqdm(range(num_samples), desc="Generating training sequences"):
            current_vector = background_activations_gpu[i]
            mask = torch.ones(num_samples, dtype=torch.bool, device=device)
            mask[i] = False
            other_vectors = background_activations_gpu[mask]
            other_labels = background_labels_gpu[mask]
            
            # Calculate ranking
            ranks = self._calculate_single_rank_k_nb(
                current_vector,
                other_vectors,
                0,
                other_labels,
                k=self.k_nb,
                device=device
            )
            
            # Store directly in Tensor
            all_sequences[i] = ranks
        
        # Save multi-dimensional Tensor
        torch.save(all_sequences, f"./experimental_results/{self.flag}/training_sequences.pt")
        return all_sequences

    def _calculate_single_rank_k_nb(self, test_vector, background_vectors, target_label, background_labels_arr, k, device):
        """
        Calculate K-NB ranking of single sample in a layer (GPU accelerated)
        Added args:
            device: Computing device (GPU/CPU)
        """
        # Ensure test vector is 2D for distance calculation (1, D)
        test_vector = test_vector.unsqueeze(0)  # From (D,) to (1, D)
        
        # Calculate Euclidean distance (PyTorch cdist, GPU accelerated)
        layer_distances = (
            (background_vectors - test_vector)  
            .norm(p=2, dim=2)  
            .permute(1, 0)  
        )
        # Sort by distance (ascending) and get indices
        sorted_indices = torch.argsort(layer_distances, dim=1)
        # Sorted background sample labels
        sorted_background_labels = background_labels_arr[sorted_indices]
        match_indices_in_sorted_tensor = torch.empty((32), device=device)
        
        for i, s in enumerate(sorted_background_labels):
            match_indices_in_sorted_tensor[i] = (torch.where(s == target_label)[0]+1)[:k].float().mean()       
        return match_indices_in_sorted_tensor