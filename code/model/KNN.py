import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend
import warnings
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

warnings.filterwarnings("ignore")

def run_knn_experiment(random_state=None, n_iter=10):
    """
    运行多次KNN实验以获得更稳定的结果
    """
    print(f"开始KNN实验，随机种子: {random_state}")
    start_time = time.time()
    
    # 加载数据
    data = pd.read_csv("/home/cyyue/Data/DTI_light/feature_con/train_tensor_KG_RDKIT.csv")
    x = data.iloc[:, :-1]
    y = np.array(data.iloc[:, -1])
    x = np.ascontiguousarray(x)
    y = np.ascontiguousarray(y)
    
    # 划分训练集和测试集
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # 数据归一化
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    
    # KNN模型参数网格
    param_grid = {
        "n_neighbors": list(range(3, 10, 2)),  # 扩展邻居范围
        "weights": ["uniform", "distance"]
    }
    
    # 使用分层K折交叉验证
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    
    # 网格搜索
    model = KNeighborsClassifier(n_jobs=-1)
    
    with parallel_backend('loky', n_jobs=-1):
        grid_search = GridSearchCV(
            estimator=model, 
            param_grid=param_grid, 
            cv=cv, 
            scoring='roc_auc',
            verbose=0,
            n_jobs=-1
        )
        grid_search.fit(xtrain, ytrain)
    
    best_model = grid_search.best_estimator_
    
    # 训练集性能
    y_pred_train = best_model.predict(xtrain)
    y_pred_train_proba = best_model.predict_proba(xtrain)[:, 1]
    
    # 测试集性能
    y_pred_test = best_model.predict(xtest)
    y_pred_test_proba = best_model.predict_proba(xtest)[:, 1]
    
    # 计算所有指标
    metrics = {
        'accuracy_train': accuracy_score(ytrain, y_pred_train),
        'precision_train': precision_score(ytrain, y_pred_train, zero_division=0),
        'recall_train': recall_score(ytrain, y_pred_train, zero_division=0),
        'f1_train': f1_score(ytrain, y_pred_train, zero_division=0),
        'auroc_train': roc_auc_score(ytrain, y_pred_train_proba),
        'aupr_train': average_precision_score(ytrain, y_pred_train_proba),
        
        'accuracy_test': accuracy_score(ytest, y_pred_test),
        'precision_test': precision_score(ytest, y_pred_test, zero_division=0),
        'recall_test': recall_score(ytest, y_pred_test, zero_division=0),
        'f1_test': f1_score(ytest, y_pred_test, zero_division=0),
        'auroc_test': roc_auc_score(ytest, y_pred_test_proba),
        'aupr_test': average_precision_score(ytest, y_pred_test_proba),
        
        'best_params': str(grid_search.best_params_),
        'best_score': grid_search.best_score_
    }
    
    return metrics, best_model, (ytest, y_pred_test, y_pred_test_proba)

def main():
    # 运行多次实验
    n_iterations = 10
    all_results = []
    all_models = []
    all_predictions = []
    
    print(f"=== 开始KNN模型评估，共 {n_iterations} 次实验 ===")
    
    for i in range(n_iterations):
        print(f"\n=== 第 {i+1}/{n_iterations} 次实验 ===")
        metrics, model, predictions = run_knn_experiment(random_state=42 + i)
        all_results.append(metrics)
        all_models.append(model)
        all_predictions.append(predictions)
        
        # 打印当前结果
        print(f"测试集 AUROC: {metrics['auroc_test']:.4f}")
        print(f"测试集 AUPR: {metrics['aupr_test']:.4f}")
        print(f"最佳参数: {metrics['best_params']}")
    
    # 转换为DataFrame
    results_df = pd.DataFrame(all_results)

    # 计算统计信息
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    
    summary = pd.DataFrame({
        'mean': results_df[numeric_cols].mean(),
        'std': results_df[numeric_cols].std(),
        'min': results_df[numeric_cols].min(),
        'max': results_df[numeric_cols].max(),
        'median': results_df[numeric_cols].median()
    })

    
    print("\n" + "="*50)
    print("=== KNN实验结果汇总 ===")
    
    print("\n测试集指标统计:")
    test_metrics = [col for col in summary.index if 'test' in col]
    for metric in test_metrics:
        mean_val = summary.loc[metric, 'mean']
        std_val = summary.loc[metric, 'std']
        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\n最佳参数分布:")
    param_counts = results_df['best_params'].value_counts()
    for params, count in param_counts.items():
        print(f"出现次数 {count}: {params}")
    
    # 保存结果
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('model'):
        os.makedirs('model')
    
    results_df.to_csv('results/knn_rdkit_experiment_results.csv', index=False)
    summary.to_csv('results/knn_rdkit_summary_statistics.csv')
    
    # 选择最佳模型（基于测试集AUROC）
    best_run_idx = results_df['auroc_test'].idxmax()
    best_model = all_models[best_run_idx]
    best_metrics = results_df.iloc[best_run_idx]
    
    # 保存最佳模型
    import joblib
    joblib.dump(best_model, 'model/KNN_RDKIT_best.pkl')
    
    # 保存最佳实验的预测结果
    ytest_best, y_pred_best, y_pred_proba_best = all_predictions[best_run_idx]
    test_results = pd.DataFrame({
        'True_Label': ytest_best,
        'Predicted_Label': y_pred_best,
        'Predicted_Probability': y_pred_proba_best
    })
    test_results.to_csv("model/KNN_RDKIT_test_predictions_best.csv", index=False)
    

if __name__ == "__main__":

    main()
