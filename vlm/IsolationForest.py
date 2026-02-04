import torch
import random
import math
import numpy as np
from torch import nn
from tqdm import tqdm
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

# 欧拉常数
EULER_GAMMA = np.euler_gamma

def _average_path_length(n_samples_leaf):
    """计算叶节点中样本的平均路径长度，与scikit-learn实现完全一致"""
    if isinstance(n_samples_leaf, (int, float)):
        n_samples_leaf = np.array([n_samples_leaf])
    elif isinstance(n_samples_leaf, torch.Tensor):
        n_samples_leaf = n_samples_leaf.cpu().numpy()
    
    n_samples_leaf = np.asarray(n_samples_leaf)
    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.0
    average_path_length[mask_2] = 1.0
    average_path_length[not_mask] = (
        2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + EULER_GAMMA)
        - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)

class PyTorchIsolationForest(nn.Module):
    """
    精确对齐scikit-learn的PyTorch孤立森林实现
    基于scikit-learn 1.2+版本的IsolationForest源码
    """
    def __init__(self, 
                 n_estimators=100, 
                 max_samples='auto', 
                 contamination='auto',
                 max_features=1.0, 
                 bootstrap=False, 
                 random_state=None,
                 verbose=0, 
                 warm_start=False):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        
        # 模型参数
        self.estimators_ = []  # 存储所有树
        self.estimators_features_ = []  # 每棵树使用的特征
        self.estimators_samples_ = []  # 每棵树使用的样本
        self.max_samples_ = None  # 实际使用的样本数
        self.offset_ = None  # 决策函数的偏移量
        self.n_features_in_ = None  # 特征数量
        self._max_features = None  # 每棵树使用的特征数
        self._average_path_length_per_tree = []  # 每棵树的平均路径长度
        self._decision_path_lengths = []  # 每棵树的决策路径长度
        self.feature_importances_ = None  # 特征重要性
        
        # 初始化随机数生成器
        self.random_state_ = check_random_state(random_state)

    def fit(self, X, y=None, sample_weight=None):
        """训练模型，X为PyTorch张量，形状为[n_samples, n_features]"""
        # 检查输入
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        
        if X.dim() != 2:
            raise ValueError(f"输入必须是二维张量，当前维度: {X.dim()}")
        
        self.n_samples_, self.n_features_in_ = X.shape
        
        # 确定每棵树的样本数
        if self.max_samples == 'auto':
            self.max_samples_ = min(256, self.n_samples_)
        elif isinstance(self.max_samples, int):
            if self.max_samples > self.n_samples_:
                print(f"警告: max_samples ({self.max_samples}) 大于样本总数 ({self.n_samples_})，将使用所有样本")
                self.max_samples_ = self.n_samples_
            else:
                self.max_samples_ = self.max_samples
        else:  # 浮点数
            self.max_samples_ = int(self.max_samples * self.n_samples_)
        
        self.max_samples_ = max(1, self.max_samples_)
        
        # 确定每棵树使用的特征数
        if isinstance(self.max_features, int):
            self._max_features = self.max_features
        else:  # 浮点数
            self._max_features = max(1, int(self.max_features * self.n_features_in_))
        
        # 树的最大深度
        max_depth = int(np.ceil(np.log2(max(self.max_samples_, 2))))
        
        # 如果不是warm start，清空已有树
        if not self.warm_start or len(self.estimators_) == 0:
            self.estimators_ = []
            self.estimators_features_ = []
            self.estimators_samples_ = []
            self._average_path_length_per_tree = []
            self._decision_path_lengths = []
        
        # 需要新增的树数量
        n_more_estimators = self.n_estimators - len(self.estimators_)
        if n_more_estimators <= 0:
            return self
        
        # 构建新的树
        for _ in tqdm(range(n_more_estimators), desc="构建孤立树", disable=self.verbose == 0):
            # 采样样本
            if self.bootstrap:
                # 有放回采样
                sample_indices = self.random_state_.choice(
                    self.n_samples_, size=self.max_samples_, replace=True)
            else:
                # 无放回采样
                sample_indices = self.random_state_.choice(
                    self.n_samples_, size=self.max_samples_, replace=False)
            
            sample_indices = torch.tensor(sample_indices, device=X.device)
            X_sample = X[sample_indices]
            
            # 采样特征
            if self._max_features < self.n_features_in_:
                feature_indices = self.random_state_.choice(
                    self.n_features_in_, size=self._max_features, replace=False)
                feature_indices = torch.tensor(feature_indices, device=X.device)
            else:
                feature_indices = torch.arange(self.n_features_in_, device=X.device)
            
            # 构建树
            tree = self._build_tree(X_sample, max_depth, feature_indices)
            self.estimators_.append(tree)
            self.estimators_features_.append(feature_indices)
            self.estimators_samples_.append(sample_indices)
            
            # 计算树的平均路径长度和决策路径长度
            node_samples = tree['node_samples']
            avg_path_length = _average_path_length(node_samples)
            decision_path_lengths = tree['node_depths']
            
            self._average_path_length_per_tree.append(avg_path_length)
            self._decision_path_lengths.append(decision_path_lengths)
        
        # 计算特征重要性
        self._compute_feature_importances()
        
        # 计算偏移量（threshold）
        if self.contamination == 'auto':
            # 原始论文中使用-0.5作为偏移量
            self.offset_ = -0.5
        else:
            # 根据污染率计算阈值
            scores = self.score_samples(X)
            self.offset_ = np.percentile(scores.cpu().numpy(), 100.0 * self.contamination)
        
        return self

    def _build_tree(self, X, max_depth, feature_indices):
        """构建单棵孤立树，返回树结构和节点信息"""
        n_samples, n_features = X.shape
        
        # 初始化根节点
        root = {
            'left': None,
            'right': None,
            'feature': None,
            'threshold': None,
            'is_leaf': False,
            'node_id': 0,
            'depth': 0
        }
        
        # 用于记录每个节点的样本数和深度
        node_samples = []
        node_depths = []
        node_queue = [root]
        next_node_id = 1
        
        while node_queue:
            node = node_queue.pop(0)
            current_depth = node['depth']
            current_node_id = node['node_id']
            
            # 确定当前节点的样本数
            if current_node_id == 0:
                # 根节点包含所有样本
                current_samples_mask = torch.ones(n_samples, dtype=torch.bool, device=X.device)
                current_samples_count = n_samples
            else:
                current_samples_count = node['sample_count']
            
            # 记录节点信息
            node_samples.append(current_samples_count)
            node_depths.append(current_depth)
            
            # 检查是否为叶子节点
            if current_samples_count <= 1 or current_depth >= max_depth:
                node['is_leaf'] = True
                continue
            
            # 随机选择一个特征
            if len(feature_indices) < self.n_features_in_:
                # 如果使用了特征采样，从采样的特征中选择
                feature_idx = self.random_state_.choice(len(feature_indices))
                feature_idx = feature_indices[feature_idx].item()
            else:
                # 否则从所有特征中选择
                feature_idx = self.random_state_.choice(self.n_features_in_)
            
            node['feature'] = feature_idx
            
            # 获取该特征的所有值
            feature_vals = X[:, feature_idx]
            
            # 找到唯一值并排序
            unique_vals = torch.unique(feature_vals)
            
            # 如果所有值都相同，无法分裂，设为叶子节点
            if len(unique_vals) == 1:
                node['is_leaf'] = True
                continue
            
            # 随机选择分裂点
            if len(unique_vals) == 2:
                # 只有两个不同值，取中间点
                split_val = (unique_vals[0] + unique_vals[1]) / 2
            else:
                # 随机选择一个值作为分裂点
                split_pos = self.random_state_.choice(len(unique_vals) - 1)
                split_val = (unique_vals[split_pos] + unique_vals[split_pos + 1]) / 2
            
            node['threshold'] = split_val.item()
            
            # 分裂样本
            if current_node_id == 0:
                left_mask = feature_vals < split_val
                right_mask = ~left_mask
            else:
                # 对于非根节点，使用父节点的掩码
                parent_mask = node['parent_mask']
                left_mask = parent_mask & (feature_vals < split_val)
                right_mask = parent_mask & ~left_mask
            
            left_count = left_mask.sum().item()
            right_count = right_mask.sum().item()
            
            # 创建左右子节点
            node['left'] = {
                'left': None,
                'right': None,
                'feature': None,
                'threshold': None,
                'is_leaf': False,
                'node_id': next_node_id,
                'depth': current_depth + 1,
                'sample_count': left_count,
                'parent_mask': left_mask if current_node_id == 0 else left_mask
            }
            next_node_id += 1
            
            node['right'] = {
                'left': None,
                'right': None,
                'feature': None,
                'threshold': None,
                'is_leaf': False,
                'node_id': next_node_id,
                'depth': current_depth + 1,
                'sample_count': right_count,
                'parent_mask': right_mask if current_node_id == 0 else right_mask
            }
            next_node_id += 1
            
            # 将子节点加入队列
            node_queue.append(node['left'])
            node_queue.append(node['right'])
        
        return {
            'tree': root,
            'node_samples': np.array(node_samples),
            'node_depths': np.array(node_depths)
        }

    def _compute_feature_importances(self):
        """计算特征重要性，基于特征被用于分裂的频率"""
        if self.n_features_in_ is None:
            return
        
        feature_counts = torch.zeros(self.n_features_in_)
        
        for tree in self.estimators_:
            # 遍历树计算每个特征的分裂次数
            queue = [tree['tree']]
            while queue:
                node = queue.pop(0)
                if node['is_leaf']:
                    continue
                
                # 记录特征使用次数
                feature_counts[node['feature']] += 1
                
                if node['left']:
                    queue.append(node['left'])
                if node['right']:
                    queue.append(node['right'])
        
        # 归一化特征重要性
        self.feature_importances_ = feature_counts.numpy() / feature_counts.sum().numpy()

    def _apply_tree(self, x, tree_struct, features):
        """将单个样本应用于树，返回叶子节点索引"""
        node = tree_struct['tree']
        
        while not node['is_leaf']:
            feature = node['feature']
            # 如果使用了特征采样，需要映射到原始特征索引
            if features is not None and len(features) < self.n_features_in_:
                # 找到特征在采样特征中的位置
                feature_pos = torch.where(features == feature)[0]
                if len(feature_pos) == 0:
                    # 不应该发生
                    return 0
                feature_pos = feature_pos.item()
                val = x[feature_pos]
            else:
                val = x[feature]
            
            if val < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        
        return node['node_id']

    def decision_function(self, X):
        """计算决策函数，值小于0的为异常"""
        check_is_fitted(self)
        return self.score_samples(X)

    def score_samples(self, X):
        """计算样本的异常分数，值越低越可能是异常"""
        check_is_fitted(self)
        
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.estimators_[0]['tree']['node_id'])
        
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        n_samples = X.shape[0]
        depths = torch.zeros(n_samples, device=X.device)
        
        # 计算平均路径长度
        average_path_length_max_samples = _average_path_length(self.max_samples_)[0]
        
        # 遍历每棵树
        for tree_idx, (tree, features) in enumerate(zip(self.estimators_, self.estimators_features_)):
            # 对每个样本计算路径长度
            for i in range(n_samples):
                x = X[i]
                leaf_idx = self._apply_tree(x, tree, features)
                
                # 累加路径长度
                depths[i] += (
                    tree['node_depths'][leaf_idx] + 
                    self._average_path_length_per_tree[tree_idx][leaf_idx] - 
                    1.0
                )
        
        # 计算最终分数 - 修复除法参数问题
        denominator = len(self.estimators_) * average_path_length_max_samples
        
        # 兼容旧版本PyTorch的除法写法
        if denominator != 0:
            depth_div = depths / denominator
        else:
            depth_div = torch.ones_like(depths)
        
        scores = 2.0 **(-depth_div)
        
        # 返回负分数，与scikit-learn保持一致（值越低越异常）
        return scores

    def predict(self, X):
        """预测样本是否为异常，1为正常，-1为异常"""
        decision_func = self.decision_function(X)
        return torch.where(decision_func < 0, -1, 1)
    import torch
import random
import math
import numpy as np
from torch import nn
from tqdm import tqdm
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

# Euler-Mascheroni constant
EULER_GAMMA = np.euler_gamma

def _average_path_length(n_samples_leaf):
    """Calculate average path length of samples in leaf nodes (exact match to scikit-learn implementation)"""
    if isinstance(n_samples_leaf, (int, float)):
        n_samples_leaf = np.array([n_samples_leaf])
    elif isinstance(n_samples_leaf, torch.Tensor):
        n_samples_leaf = n_samples_leaf.cpu().numpy()
    
    n_samples_leaf = np.asarray(n_samples_leaf)
    n_samples_leaf_shape = n_samples_leaf.shape
    n_samples_leaf = n_samples_leaf.reshape((1, -1))
    average_path_length = np.zeros(n_samples_leaf.shape)

    mask_1 = n_samples_leaf <= 1
    mask_2 = n_samples_leaf == 2
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.0
    average_path_length[mask_2] = 1.0
    average_path_length[not_mask] = (
        2.0 * (np.log(n_samples_leaf[not_mask] - 1.0) + EULER_GAMMA)
        - 2.0 * (n_samples_leaf[not_mask] - 1.0) / n_samples_leaf[not_mask]
    )

    return average_path_length.reshape(n_samples_leaf_shape)

class PyTorchIsolationForest(nn.Module):
    """
    PyTorch implementation of IsolationForest with exact alignment to scikit-learn
    Based on scikit-learn 1.2+ IsolationForest source code
    """
    def __init__(self, 
                 n_estimators=100, 
                 max_samples='auto', 
                 contamination='auto',
                 max_features=1.0, 
                 bootstrap=False, 
                 random_state=None,
                 verbose=0, 
                 warm_start=False):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        
        # Model parameters
        self.estimators_ = []  # Store all isolation trees
        self.estimators_features_ = []  # Features used by each tree
        self.estimators_samples_ = []  # Samples used by each tree
        self.max_samples_ = None  # Actual number of samples used
        self.offset_ = None  # Offset for decision function
        self.n_features_in_ = None  # Number of input features
        self._max_features = None  # Number of features used per tree
        self._average_path_length_per_tree = []  # Average path length for each tree
        self._decision_path_lengths = []  # Decision path lengths for each tree
        self.feature_importances_ = None  # Feature importance scores
        
        # Initialize random number generator
        self.random_state_ = check_random_state(random_state)

    def fit(self, X, y=None, sample_weight=None):
        """Train the model with PyTorch tensor input
        Args:
            X: PyTorch tensor with shape [n_samples, n_features]
            y: Ignored (compatibility with scikit-learn API)
            sample_weight: Ignored (compatibility with scikit-learn API)
        Returns:
            self: Trained model instance
        """
        # Input validation
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        
        if X.dim() != 2:
            raise ValueError(f"Input must be 2D tensor, current dimensions: {X.dim()}")
        
        self.n_samples_, self.n_features_in_ = X.shape
        
        # Determine number of samples per tree
        if self.max_samples == 'auto':
            self.max_samples_ = min(256, self.n_samples_)
        elif isinstance(self.max_samples, int):
            if self.max_samples > self.n_samples_:
                print(f"Warning: max_samples ({self.max_samples}) > total samples ({self.n_samples_}), using all samples")
                self.max_samples_ = self.n_samples_
            else:
                self.max_samples_ = self.max_samples
        else:  # Float value (fraction of total samples)
            self.max_samples_ = int(self.max_samples * self.n_samples_)
        
        self.max_samples_ = max(1, self.max_samples_)
        
        # Determine number of features per tree
        if isinstance(self.max_features, int):
            self._max_features = self.max_features
        else:  # Float value (fraction of total features)
            self._max_features = max(1, int(self.max_features * self.n_features_in_))
        
        # Maximum depth of trees
        max_depth = int(np.ceil(np.log2(max(self.max_samples_, 2))))
        
        # Clear existing trees if not warm start
        if not self.warm_start or len(self.estimators_) == 0:
            self.estimators_ = []
            self.estimators_features_ = []
            self.estimators_samples_ = []
            self._average_path_length_per_tree = []
            self._decision_path_lengths = []
        
        # Number of additional trees to build
        n_more_estimators = self.n_estimators - len(self.estimators_)
        if n_more_estimators <= 0:
            return self
        
        # Build new trees
        for _ in tqdm(range(n_more_estimators), desc="Building isolation trees", disable=self.verbose == 0):
            # Sample samples
            if self.bootstrap:
                # Sampling with replacement
                sample_indices = self.random_state_.choice(
                    self.n_samples_, size=self.max_samples_, replace=True)
            else:
                # Sampling without replacement
                sample_indices = self.random_state_.choice(
                    self.n_samples_, size=self.max_samples_, replace=False)
            
            sample_indices = torch.tensor(sample_indices, device=X.device)
            X_sample = X[sample_indices]
            
            # Sample features
            if self._max_features < self.n_features_in_:
                feature_indices = self.random_state_.choice(
                    self.n_features_in_, size=self._max_features, replace=False)
                feature_indices = torch.tensor(feature_indices, device=X.device)
            else:
                feature_indices = torch.arange(self.n_features_in_, device=X.device)
            
            # Build tree
            tree = self._build_tree(X_sample, max_depth, feature_indices)
            self.estimators_.append(tree)
            self.estimators_features_.append(feature_indices)
            self.estimators_samples_.append(sample_indices)
            
            # Calculate average path length and decision path lengths for the tree
            node_samples = tree['node_samples']
            avg_path_length = _average_path_length(node_samples)
            decision_path_lengths = tree['node_depths']
            
            self._average_path_length_per_tree.append(avg_path_length)
            self._decision_path_lengths.append(decision_path_lengths)
        
        # Calculate feature importances
        self._compute_feature_importances()
        
        # Calculate offset (threshold)
        if self.contamination == 'auto':
            # Use -0.5 as offset (original paper recommendation)
            self.offset_ = -0.5
        else:
            # Calculate threshold based on contamination rate
            scores = self.score_samples(X)
            self.offset_ = np.percentile(scores.cpu().numpy(), 100.0 * self.contamination)
        
        return self

    def _build_tree(self, X, max_depth, feature_indices):
        """Build a single isolation tree and return tree structure with node information
        Args:
            X: Sample tensor for tree building [n_samples, n_features]
            max_depth: Maximum depth of the tree
            feature_indices: Indices of features to use for splitting
        Returns:
            dict: Tree structure with node samples and depths
        """
        n_samples, n_features = X.shape
        
        # Initialize root node
        root = {
            'left': None,
            'right': None,
            'feature': None,
            'threshold': None,
            'is_leaf': False,
            'node_id': 0,
            'depth': 0
        }
        
        # Track sample count and depth for each node
        node_samples = []
        node_depths = []
        node_queue = [root]
        next_node_id = 1
        
        while node_queue:
            node = node_queue.pop(0)
            current_depth = node['depth']
            current_node_id = node['node_id']
            
            # Determine sample count for current node
            if current_node_id == 0:
                # Root node contains all samples
                current_samples_mask = torch.ones(n_samples, dtype=torch.bool, device=X.device)
                current_samples_count = n_samples
            else:
                current_samples_count = node['sample_count']
            
            # Record node information
            node_samples.append(current_samples_count)
            node_depths.append(current_depth)
            
            # Check if node is leaf
            if current_samples_count <= 1 or current_depth >= max_depth:
                node['is_leaf'] = True
                continue
            
            # Randomly select a feature for splitting
            if len(feature_indices) < self.n_features_in_:
                # Select from sampled features
                feature_idx = self.random_state_.choice(len(feature_indices))
                feature_idx = feature_indices[feature_idx].item()
            else:
                # Select from all features
                feature_idx = self.random_state_.choice(self.n_features_in_)
            
            node['feature'] = feature_idx
            
            # Get all values of the selected feature
            feature_vals = X[:, feature_idx]
            
            # Find unique values and sort
            unique_vals = torch.unique(feature_vals)
            
            # If all values are the same, cannot split - mark as leaf
            if len(unique_vals) == 1:
                node['is_leaf'] = True
                continue
            
            # Randomly select split point
            if len(unique_vals) == 2:
                # Only two distinct values - take midpoint
                split_val = (unique_vals[0] + unique_vals[1]) / 2
            else:
                # Randomly select split position between unique values
                split_pos = self.random_state_.choice(len(unique_vals) - 1)
                split_val = (unique_vals[split_pos] + unique_vals[split_pos + 1]) / 2
            
            node['threshold'] = split_val.item()
            
            # Split samples
            if current_node_id == 0:
                left_mask = feature_vals < split_val
                right_mask = ~left_mask
            else:
                # For non-root nodes, use parent mask
                parent_mask = node['parent_mask']
                left_mask = parent_mask & (feature_vals < split_val)
                right_mask = parent_mask & ~left_mask
            
            left_count = left_mask.sum().item()
            right_count = right_mask.sum().item()
            
            # Create left and right child nodes
            node['left'] = {
                'left': None,
                'right': None,
                'feature': None,
                'threshold': None,
                'is_leaf': False,
                'node_id': next_node_id,
                'depth': current_depth + 1,
                'sample_count': left_count,
                'parent_mask': left_mask if current_node_id == 0 else left_mask
            }
            next_node_id += 1
            
            node['right'] = {
                'left': None,
                'right': None,
                'feature': None,
                'threshold': None,
                'is_leaf': False,
                'node_id': next_node_id,
                'depth': current_depth + 1,
                'sample_count': right_count,
                'parent_mask': right_mask if current_node_id == 0 else right_mask
            }
            next_node_id += 1
            
            # Add child nodes to queue
            node_queue.append(node['left'])
            node_queue.append(node['right'])
        
        return {
            'tree': root,
            'node_samples': np.array(node_samples),
            'node_depths': np.array(node_depths)
        }

    def _compute_feature_importances(self):
        """Calculate feature importances based on split frequency
        Feature importance = number of times feature is used for splitting / total splits
        """
        if self.n_features_in_ is None:
            return
        
        feature_counts = torch.zeros(self.n_features_in_)
        
        for tree in self.estimators_:
            # Traverse tree to count feature splits
            queue = [tree['tree']]
            while queue:
                node = queue.pop(0)
                if node['is_leaf']:
                    continue
                
                # Record feature usage
                feature_counts[node['feature']] += 1
                
                if node['left']:
                    queue.append(node['left'])
                if node['right']:
                    queue.append(node['right'])
        
        # Normalize feature importances
        self.feature_importances_ = feature_counts.numpy() / feature_counts.sum().numpy()

    def _apply_tree(self, x, tree_struct, features):
        """Apply single sample to tree and return leaf node index
        Args:
            x: Single sample tensor [n_features]
            tree_struct: Tree structure dictionary
            features: Feature indices used by the tree
        Returns:
            int: Leaf node index
        """
        node = tree_struct['tree']
        
        while not node['is_leaf']:
            feature = node['feature']
            # Map to original feature index if feature sampling was used
            if features is not None and len(features) < self.n_features_in_:
                # Find position of feature in sampled features
                feature_pos = torch.where(features == feature)[0]
                if len(feature_pos) == 0:
                    # Should not happen
                    return 0
                feature_pos = feature_pos.item()
                val = x[feature_pos]
            else:
                val = x[feature]
            
            if val < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        
        return node['node_id']

    def decision_function(self, X):
        """Compute decision function (values < 0 indicate anomalies)
        Args:
            X: Input tensor [n_samples, n_features]
        Returns:
            torch.Tensor: Decision scores (lower = more anomalous)
        """
        check_is_fitted(self)
        return self.score_samples(X)

    def score_samples(self, X):
        """Compute anomaly scores for samples (lower scores = more anomalous)
        Args:
            X: Input tensor [n_samples, n_features]
        Returns:
            torch.Tensor: Anomaly scores
        """
        check_is_fitted(self)
        
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.estimators_[0]['tree']['node_id'])
        
        if X.dim() == 1:
            X = X.unsqueeze(0)
        
        n_samples = X.shape[0]
        depths = torch.zeros(n_samples, device=X.device)
        
        # Calculate average path length for max samples
        average_path_length_max_samples = _average_path_length(self.max_samples_)[0]
        
        # Iterate over all trees
        for tree_idx, (tree, features) in enumerate(zip(self.estimators_, self.estimators_features_)):
            # Calculate path length for each sample
            for i in range(n_samples):
                x = X[i]
                leaf_idx = self._apply_tree(x, tree, features)
                
                # Accumulate path lengths
                depths[i] += (
                    tree['node_depths'][leaf_idx] + 
                    self._average_path_length_per_tree[tree_idx][leaf_idx] - 
                    1.0
                )
        
        # Calculate final scores - fix division parameter issue
        denominator = len(self.estimators_) * average_path_length_max_samples
        
        # Compatible division for older PyTorch versions
        if denominator != 0:
            depth_div = depths / denominator
        else:
            depth_div = torch.ones_like(depths)
        
        scores = 2.0 **(-depth_div)
        
        # Return negative scores (consistent with scikit-learn: lower = more anomalous)
        return scores

    def predict(self, X):
        """Predict if samples are anomalies (1 = normal, -1 = anomalous)
        Args:
            X: Input tensor [n_samples, n_features]
        Returns:
            torch.Tensor: Prediction labels (1/-1)
        """
        decision_func = self.decision_function(X)
        return torch.where(decision_func < 0, -1, 1)