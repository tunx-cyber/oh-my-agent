'''
先在高层图中快速找到大致方向，再在底层图中精确定位目标。
'''
import numpy as np
import random
from collections import defaultdict
from sklearn.metrics.pairwise import euclidean_distances
class SimpleHNSW:
    """简化的HNSW实现，用于学习演示"""
    
    def __init__(self, max_elements=1000, M=10, ef_construction=50, max_layers=6):
        """
        初始化HNSW索引
        
        参数:
        - max_elements: 最大元素数量
        - M: 每个节点的最大连接数
        - ef_construction: 构建时的搜索范围
        - max_layers: 最大层数
        """
        self.max_elements = max_elements
        self.M = M  # 每个节点的最大连接数
        self.ef_construction = ef_construction  # 构建时的搜索范围
        self.max_layers = max_layers  # 最大层数
        
        # 存储所有数据点
        self.data_points = []
        # 每层的图结构（邻接表），每层是一个字典，key是节点ID，value是邻居列表
        self.layers = [defaultdict(list) for _ in range(max_layers)]
        # 全局入口点（最高层的节点）
        self.entry_point = None
        self.entry_level = -1  # 入口点所在的最高层级
        
    def _random_level(self):
        """随机生成节点的层级（指数分布）"""
        level = 0
        while random.random() < 0.5 and level < self.max_layers - 1:
            level += 1
        return level
    
    def _euclidean_distance(self, a, b):
        """计算欧氏距离"""
        return np.sqrt(np.sum((a - b) ** 2))
    
    def _search_layer(self, query, entry_point, ef, layer):
        """
        在指定层搜索最近邻
        
        参数:
        - query: 查询向量
        - entry_point: 搜索起始点
        - ef: 搜索范围（返回的候选点数量）
        - layer: 搜索的层级
        """
        if entry_point is None or entry_point not in self.layers[layer]:
            return []
            
        visited = set([entry_point])
        # 候选集：存储(距离, 节点ID)元组，从入口点开始
        candidates = [(self._euclidean_distance(query, self.data_points[entry_point]), entry_point)]
        # 使用堆来维护候选集（这里简化为列表排序）
        results = []
        
        while candidates and len(results) < ef:
            # 获取距离最近的候选点
            candidates.sort(key=lambda x: x[0])
            current_dist, current_point = candidates.pop(0)
            
            # 检查是否应该将当前点加入结果
            if not results or current_dist < results[-1][0]:
                results.append((current_dist, current_point))
                results.sort(key=lambda x: x[0])  # 保持结果按距离排序
                if len(results) > ef:
                    results = results[:ef]  # 限制结果集大小
            
            # 探索当前点的所有邻居节点
            for neighbor in self.layers[layer][current_point]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist = self._euclidean_distance(query, self.data_points[neighbor])
                    candidates.append((dist, neighbor))
        
        return results
    
    def add_point(self, point):
        """
        向HNSW中添加新点
        
        参数:
        - point: 要添加的数据点向量
        """
        if len(self.data_points) >= self.max_elements:
            raise ValueError("达到最大容量")
        
        point_id = len(self.data_points)  # 新点在data_points中的id索引
        self.data_points.append(point)
        
        # 确定新点的层级
        level = self._random_level()
        
        # 如果是第一个点，设为入口点
        if self.entry_point is None:
            self.entry_point = point_id
            self.entry_level = level
            for l in range(level + 1):
                self.layers[l][point_id] = []  # 在新点的每一层创建空邻居列表
            return
        
        # 从最高层开始搜索，找到每层的最近邻
        current_point = self.entry_point
        current_max_level = self.entry_level
        
        # 从顶层开始搜索，找到每层的入口点
        for l in range(current_max_level, level, -1):
            if l < len(self.layers):
                results = self._search_layer(point, current_point, 1, l)
                if results:
                    current_point = results[0][1]  # 更新为最近的点
        
        # 从新点的最高层开始，逐层向下插入并建立连接
        for l in range(min(level, current_max_level), -1, -1):
            # 在当前层搜索ef_construction个最近邻
            results = self._search_layer(point, current_point, self.ef_construction, l)
            
            # 选择前M个最近邻作为连接
            neighbors = [idx for _, idx in results[:self.M]]
            
            # 在新点的当前层创建连接
            self.layers[l][point_id] = neighbors.copy()
            
            # 双向连接：邻居也连接到新点
            for neighbor in neighbors:
                if len(self.layers[l][neighbor]) < self.M:
                    # 邻居连接数未满，直接添加
                    self.layers[l][neighbor].append(point_id)
                else:
                    # 如果邻居连接数已满，替换最远的连接
                    neighbor_neighbors = self.layers[l][neighbor]
                    distances = [self._euclidean_distance(self.data_points[neighbor], 
                                                         self.data_points[n]) for n in neighbor_neighbors]
                    max_idx = np.argmax(distances)  # 找到最远的邻居
                    # 如果新点更近，则替换最远的邻居
                    if self._euclidean_distance(self.data_points[neighbor], point) < distances[max_idx]:
                        neighbor_neighbors[max_idx] = point_id
            
            # 更新当前点用于下一层的搜索
            if results:
                current_point = results[0][1]
        
        # 如果新点的层级比当前入口点高，更新入口点
        if level > self.entry_level:
            self.entry_point = point_id
            self.entry_level = level
    
    def search(self, query, k=5, ef_search=50):
        """
        在HNSW中搜索最近邻
        
        参数:
        - query: 查询向量
        - k: 返回的最近邻数量
        - ef_search: 搜索时的候选集大小（越大精度越高但速度越慢）
        
        返回:
        - 包含(节点ID, 距离)的列表，按距离升序排列
        """
        if self.entry_point is None:
            return []
        
        current_point = self.entry_point
        current_level = self.entry_level
        
        # 从顶层开始搜索
        for l in range(current_level, 0, -1):
            results = self._search_layer(query, current_point, 1, l)
            if results:
                current_point = results[0][1]  # 更新为每层的入口点
        
        # 在最底层进行精细搜索
        results = self._search_layer(query, current_point, ef_search, 0)
        
        # 返回前k个结果
        return [(idx, dist) for dist, idx in results[:k]]
    
'''
episode_2_RAG.ANN.IVF 的 Docstring
'''
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from collections import defaultdict
import time

np.random.seed(42)

def generate_sample_data(n_samples=1000, dim=2):
    """生成示例数据：三个明显分离的高斯分布簇"""
    # 第一个簇
    cluster1 = np.random.normal(loc=[2, 2], scale=0.5, size=(n_samples//3, dim))
    # 第二个簇  
    cluster2 = np.random.normal(loc=[8, 3], scale=0.6, size=(n_samples//3, dim))
    # 第三个簇
    cluster3 = np.random.normal(loc=[5, 8], scale=0.4, size=(n_samples - 2*(n_samples//3), dim))
    
    data = np.vstack([cluster1, cluster2, cluster3])
    return data

# 生成数据
data = generate_sample_data()
print(f"数据形状: {data.shape}")

class SimpleKMeans:
    """简化的K-means实现用于IVF聚类"""
    
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.labels_ = None
    
    def fit(self, X):
        n_samples, n_features = X.shape
        
        # 1. 随机初始化质心
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]
        
        for iteration in range(self.max_iters):
            # 2. 分配每个点到最近的质心
            distances = euclidean_distances(X, self.centroids)
            labels = np.argmin(distances, axis=1)
            
            # 3. 更新质心位置
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            
            # 检查收敛
            if np.allclose(self.centroids, new_centroids):
                break
                
            self.centroids = new_centroids
        
        self.labels_ = labels
        return self
    
class SimpleIVF:
    """简化的IVF实现"""
    
    def __init__(self, n_clusters=3, n_probe=2):
        self.n_clusters = n_clusters
        self.n_probe = n_probe  # 搜索时探测的簇数量
        self.kmeans = None
        self.inverted_lists = None  # 倒排列表
        self.centroids = None
        self.is_trained = False
    
    def train(self, data):
        """训练IVF索引：对数据进行聚类"""
        print("开始训练IVF索引...")
        self.kmeans = SimpleKMeans(n_clusters=self.n_clusters)
        self.kmeans.fit(data)
        self.centroids = self.kmeans.centroids
        self.is_trained = True
        print(f"训练完成，得到{self.n_clusters}个簇")
    
    def build_index(self, data):
        """构建倒排索引"""
        if not self.is_trained:
            self.train(data)
        
        # 初始化倒排列表
        self.inverted_lists = defaultdict(list)
        
        # 将每个向量分配到最近的簇
        distances = euclidean_distances(data, self.centroids)
        labels = np.argmin(distances, axis=1)
        
        # 构建倒排列表：簇ID -> 该簇中所有向量的索引
        for idx, label in enumerate(labels):
            self.inverted_lists[label].append(idx)
        
        print("倒排索引构建完成:")
        for cluster_id, items in self.inverted_lists.items():
            print(f"  簇{cluster_id}: {len(items)}个向量")
    
    def search(self, query, k=5, data=None):
        """IVF搜索：先找最近的簇，然后在簇内搜索"""
        if data is None:
            data = self.data
            
        # 1. 粗略搜索：找到最近的n_probe个簇
        distances_to_centroids = euclidean_distances([query], self.centroids)[0]
        nearest_cluster_indices = np.argsort(distances_to_centroids)[:self.n_probe]
        
        # 2. 精细搜索：在选中的簇内进行暴力搜索
        candidate_indices = []
        for cluster_idx in nearest_cluster_indices:
            candidate_indices.extend(self.inverted_lists[cluster_idx])
        
        if not candidate_indices:
            return [], []
        
        # 在候选向量中计算距离
        candidate_vectors = data[candidate_indices]
        distances = euclidean_distances([query], candidate_vectors)[0]
        
        # 获取最近的k个结果
        if k > len(distances):
            k = len(distances)
            
        nearest_indices_within_candidates = np.argsort(distances)[:k]
        
        # 映射回原始索引
        final_indices = [candidate_indices[i] for i in nearest_indices_within_candidates]
        final_distances = distances[nearest_indices_within_candidates]
        
        return final_indices, final_distances
    
    def brute_force_search(self, query, k=5, data=None):
        """暴力搜索作为对比基准"""
        if data is None:
            data = self.data
            
        distances = euclidean_distances([query], data)[0]
        nearest_indices = np.argsort(distances)[:k]
        return nearest_indices, distances[nearest_indices]
    
def demonstrate_ivf():
    """完整演示IVF算法"""
    print("=" * 60)
    print("IVF算法演示")
    print("=" * 60)
    
    # 1. 生成数据
    data = generate_sample_data(300, 2)
    print(f"生成{len(data)}个二维数据点")
    
    # 2. 创建并训练IVF索引
    ivf = SimpleIVF(n_clusters=3, n_probe=2)
    ivf.data = data  # 保存数据引用
    ivf.build_index(data)
    
    # 3. 选择一个查询点
    query_point = np.array([5.0, 5.0])
    print(f"\n查询点: {query_point}")
    
    # 4. 使用IVF搜索
    start_time = time.time()
    ivf_indices, ivf_distances = ivf.search(query_point, k=5, data=data)
    ivf_time = time.time() - start_time
    
    # 5. 使用暴力搜索作为对比
    start_time = time.time()
    bf_indices, bf_distances = ivf.brute_force_search(query_point, k=5, data=data)
    bf_time = time.time() - start_time
    
    # 6. 显示结果
    print(f"\n搜索结果对比:")
    print(f"IVF搜索  - 找到{len(ivf_indices)}个最近邻, 耗时: {ivf_time:.6f}秒")
    print(f"暴力搜索 - 找到{len(bf_indices)}个最近邻, 耗时: {bf_time:.6f}秒")
    
    print(f"\n速度提升: {bf_time/ivf_time:.2f}倍")
    
    print(f"\n最近邻索引 (IVF): {ivf_indices}")
    print(f"最近邻索引 (暴力): {bf_indices}")
    
    # 7. 检查召回率
    intersection = set(ivf_indices) & set(bf_indices)
    recall = len(intersection) / len(bf_indices)
    print(f"召回率: {recall:.2%} ({len(intersection)}/{len(bf_indices)})")
    
    return ivf, data, query_point, ivf_indices, bf_indices

# 运行演示
ivf, data, query, ivf_results, bf_results = demonstrate_ivf()