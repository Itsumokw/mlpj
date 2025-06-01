import numpy as np
from scipy import stats, signal
import warnings
from typing import Union, List

# 忽略部分数学计算警告
warnings.filterwarnings('ignore', category=RuntimeWarning)

class FeatureExtractor:
    """轻量级时序特征提取器（支持30+特征）"""
    
    def __init__(self, window_size=10, feature_list: Union[str, List[str]] = 'all'):
        """
        参数:
        window_size: 滑动窗口大小
        feature_list: 选择使用的特征集，可选'all'或特征名列表
        """
        self.window_size = window_size
        self.feature_list = self._validate_features(feature_list)
        
    def _validate_features(self, feature_list):
        """验证并标准化特征列表"""
        all_features = [
            'mean', 'std', 'min', 'max', 'range', 'median', 'mad',
            'skew', 'kurtosis', 'quantile_25', 'quantile_75', 'iqr',
            'fft_amp1', 'fft_amp_mean', 'fft_phase1',
            'slope', 'intercept', 'r_value', 'p_value', 
            'zero_crossing', 'autocorr_lag1', 'autocorr_lag2',
            'entropy', 'hurst', 'lyapunov', 'binned_entropy',
            'c3', 'cid', 'mean_abs_change', 'mean_second_derivative',
            'number_peaks', 'permutation_entropy'
        ]
        
        if feature_list == 'all':
            return all_features
        elif isinstance(feature_list, list):
            valid_features = [f for f in feature_list if f in all_features]
            if not valid_features:
                raise ValueError(f"无效特征名，请从以下选择: {all_features}")
            return valid_features
        else:
            raise TypeError("feature_list 必须是 'all' 或特征名列表")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        为时序数据提取特征
        
        参数:
        X: 输入时序数据 (1D数组)
        
        返回:
        特征矩阵 (n_samples, n_features)
        """
        # 生成滑动窗口视图
        windows = self.sliding_window(X, self.window_size)
        n_windows = len(windows)
        
        # 为每个窗口计算特征
        features = []
        for i in range(n_windows):
            window = windows[i]
            window_features = []
            
            for feat_name in self.feature_list:
                # 动态调用特征计算方法
                feat_func = getattr(self, f"_calc_{feat_name}")
                value = feat_func(window)
                window_features.append(value)
                
            features.append(window_features)
            
        return np.array(features)
    
    @staticmethod
    def sliding_window(X: np.ndarray, window: int) -> np.ndarray:
        """零拷贝滑动窗口生成"""
        shape = (X.shape[0] - window + 1, window)
        strides = (X.strides[0], X.strides[0])
        return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    
    # ====================== 统计特征 ======================
    def _calc_mean(self, window):
        return np.mean(window)
    
    def _calc_std(self, window):
        return np.std(window)
    
    def _calc_min(self, window):
        return np.min(window)
    
    def _calc_max(self, window):
        return np.max(window)
    
    def _calc_range(self, window):
        return np.max(window) - np.min(window)
    
    def _calc_median(self, window):
        return np.median(window)
    
    def _calc_mad(self, window):
        """中值绝对偏差"""
        return np.median(np.abs(window - np.median(window)))
    
    def _calc_skew(self, window):
        return stats.skew(window)
    
    def _calc_kurtosis(self, window):
        return stats.kurtosis(window)
    
    def _calc_quantile_25(self, window):
        return np.quantile(window, 0.25)
    
    def _calc_quantile_75(self, window):
        return np.quantile(window, 0.75)
    
    def _calc_iqr(self, window):
        """四分位距"""
        return np.quantile(window, 0.75) - np.quantile(window, 0.25)
    
    # ====================== 频域特征 ======================
    def _calc_fft_amp1(self, window):
        """FFT基频幅值"""
        fft = np.fft.rfft(window)
        return np.abs(fft[1]) if len(fft) > 1 else 0.0
    
    def _calc_fft_amp_mean(self, window):
        """FFT平均幅值（去除直流分量）"""
        fft = np.fft.rfft(window)
        return np.mean(np.abs(fft[1:])) if len(fft) > 1 else 0.0
    
    def _calc_fft_phase1(self, window):
        """FFT基频相位"""
        fft = np.fft.rfft(window)
        return np.angle(fft[1]) if len(fft) > 1 else 0.0
    
    # ====================== 趋势特征 ======================
    def _calc_slope(self, window):
        """线性趋势斜率"""
        return np.polyfit(np.arange(len(window)), window, 1)[0]
    
    def _calc_intercept(self, window):
        """线性趋势截距"""
        return np.polyfit(np.arange(len(window)), window, 1)[1]
    
    def _calc_r_value(self, window):
        """线性拟合相关系数"""
        _, _, r_value, _, _ = stats.linregress(np.arange(len(window)), window)
        return r_value
    
    def _calc_p_value(self, window):
        """线性拟合p值"""
        _, _, _, p_value, _ = stats.linregress(np.arange(len(window)), window)
        return p_value
    
    # ====================== 时序特征 ======================
    def _calc_zero_crossing(self, window):
        """过零率"""
        return len(np.where(np.diff(np.sign(window)))[0])
    
    def _calc_autocorr_lag1(self, window):
        """滞后1自相关"""
        return np.corrcoef(window[:-1], window[1:])[0, 1] if len(window) > 1 else 0.0
    
    def _calc_autocorr_lag2(self, window):
        """滞后2自相关"""
        return np.corrcoef(window[:-2], window[2:])[0, 1] if len(window) > 2 else 0.0
    
    # ====================== 非线性特征 ======================
    def _calc_entropy(self, window):
        """近似熵（简化计算）"""
        if len(window) < 3:
            return 0.0
        diff = np.diff(window)
        return np.sum(diff * np.log(np.abs(diff) + 1e-10))
    
    def _calc_hurst(self, window):
        """Hurst指数（重标极差法）"""
        n = len(window)
        if n < 2:
            return 0.5
        
        # 计算累积偏差
        deviations = window - np.mean(window)
        cumulative_dev = np.cumsum(deviations)
        
        # 计算极差
        r = np.max(cumulative_dev) - np.min(cumulative_dev)
        
        # 计算标准差
        s = np.std(window)
        
        return np.log(r / s) / np.log(n) if s != 0 else 0.0
    
    def _calc_lyapunov(self, window):
        """Lyapunov指数（简化估计）"""
        if len(window) < 3:
            return 0.0
        diff = np.diff(window)
        return np.mean(np.log(np.abs(diff) + 1e-10))
    
    def _calc_binned_entropy(self, window, bins=10):
        """分桶熵"""
        hist, _ = np.histogram(window, bins=bins)
        prob = hist / np.sum(hist)
        return stats.entropy(prob)
    
    # ====================== 高级特征 ======================
    def _calc_c3(self, window, lag=1):
        """非线性度量C3"""
        if len(window) < 2 * lag:
            return 0.0
        return np.mean(window[2*lag:] * window[lag:-lag] * window[:-2*lag])
    
    def _calc_cid(self, window):
        """时序复杂度"""
        diff = np.diff(window)
        return np.sqrt(np.sum(diff ** 2))
    
    def _calc_mean_abs_change(self, window):
        """绝对变化均值"""
        return np.mean(np.abs(np.diff(window)))
    
    def _calc_mean_second_derivative(self, window):
        """二阶导数均值"""
        if len(window) < 3:
            return 0.0
        second_diff = np.diff(window, 2)
        return np.mean(second_diff)
    
    def _calc_number_peaks(self, window, n=3):
        """窗口内峰值数量"""
        peaks, _ = signal.find_peaks(window, prominence=0.1)
        return len(peaks)
    
    def _calc_permutation_entropy(self, window, order=3, delay=1):
        """排列熵"""
        n = len(window)
        if n < order * delay:
            return 0.0
        
        permutations = {}
        for i in range(n - (order-1)*delay):
            # 获取顺序模式
            segment = window[i:i+order*delay:delay]
            pattern = tuple(np.argsort(segment))
            permutations[pattern] = permutations.get(pattern, 0) + 1
            
        # 计算熵值
        total = sum(permutations.values())
        entropy = 0.0
        for count in permutations.values():
            p = count / total
            entropy -= p * np.log(p)
            
        return entropy / np.log(np.math.factorial(order))
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        为时序数据提取特征，返回原始数据与特征的组合

        参数:
        X: 输入时序数据 (1D数组)

        返回:
        组合后的特征矩阵 (n_samples, window_size + n_features)
        """
        # 生成滑动窗口视图
        windows = self.sliding_window(X, self.window_size)
        n_windows = len(windows)

        # 特征容器
        feature_results = []

        # 处理每个窗口
        for i in range(n_windows):
            window = windows[i]
            window_features = []
            
            # 保留原始数据
            raw_data = window.copy()
            
            # 计算特征
            for feat_name in self.feature_list:
                feat_func = getattr(self, f"_calc_{feat_name}")
                value = feat_func(window)
                window_features.append(value)
            
            # 组合原始数据和提取的特征
            combined = np.concatenate([raw_data, np.array(window_features)])
            feature_results.append(combined)
            
        return np.array(feature_results)
    
    
    def feature_size(self):
        """返回提取的特征数量"""
        return len(self.feature_list) if self.feature_list != 'all' else len(self._validate_features('all'))