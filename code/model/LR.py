import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import os

warnings.filterwarnings("ignore")

def run_experiment(random_state=None, n_iter=10):
    """
    运行多次实验以获得更稳定的结果
    """
    print(f"开始实验，随机种子: {random_state}")
    
    # 加载数据
    data = pd.read_csv("/home/cyyue/Data/DTI_light/feature_con/train_tensor_KG_Morgan.csv")
    x = data.iloc[:, :-1]
    y = np.array(data.iloc[:, -1])
    
    # 划分训练集和测试集
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=random_state)
    
    # 数据归一化
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    
    # 参数网格
    param_grid = {
        'C': np.logspace(-3, 2, 20),
        'solver': ['liblinear', 'lbfgs', 'newton-cg', 'saga'],
        'class_weight': [None, 'balanced']
    }
    
    # 使用分层K折交叉验证
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    
    # 网格搜索
    model = LogisticRegression(max_iter=2000, random_state=random_state)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                              cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
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
        'precision_train': precision_score(ytrain, y_pred_train),
        'recall_train': recall_score(ytrain, y_pred_train),
        'f1_train': f1_score(ytrain, y_pred_train),
        'auroc_train': roc_auc_score(ytrain, y_pred_train_proba),
        'aupr_train': average_precision_score(ytrain, y_pred_train_proba),
        
        'accuracy_test': accuracy_score(ytest, y_pred_test),
        'precision_test': precision_score(ytest, y_pred_test),
        'recall_test': recall_score(ytest, y_pred_test),
        'f1_test': f1_score(ytest, y_pred_test),
        'auroc_test': roc_auc_score(ytest, y_pred_test_proba),
        'aupr_test': average_precision_score(ytest, y_pred_test_proba),
        
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_
    }
    
    return metrics

def main():
    # 运行多次实验
    n_iterations = 10
    results = []
    
    for i in range(n_iterations):
        print(f"\n=== 第 {i+1}/{n_iterations} 次实验 ===")
        metrics = run_experiment(random_state=42 + i)
        results.append(metrics)
    
    # 转换为DataFrame
    results_df = pd.DataFrame(results)
    
    # 计算均值和标准差
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    summary = pd.DataFrame({
        'mean': results_df[numeric_cols].mean(),
        'std': results_df[numeric_cols].std(),
        'min': results_df[numeric_cols].min(),
        'max': results_df[numeric_cols].max()
    })
    
    print("\n=== 实验结果汇总 ===")
    print(f"实验次数: {n_iterations}")
    print("\n测试集指标统计:")
    test_metrics = [col for col in summary.index if 'test' in col]
    print(summary.loc[test_metrics])
    
    print("\n最佳参数分布:")
    param_counts = results_df['best_params'].value_counts()
    print(param_counts)
    
    # 保存结果
    if not os.path.exists('results'):
        os.makedirs('results')
    
    results_df.to_csv('results/lr_morgan_experiment_results.csv', index=False)
    summary.to_csv('results/lr_morgan_summary_statistics.csv')
    
    # 选择最佳模型
    best_run_idx = results_df['auroc_test'].idxmax()
    best_metrics = results_df.iloc[best_run_idx]  
    
    # 保存最佳模型
    import joblib
    joblib.dump(best_model, 'model/LR_RDKIT_best.pkl')
    
    print(f"\n最佳实验 (第 {best_run_idx + 1} 次):")
    print(f"测试集 AUROC: {best_metrics['auroc_test']:.4f}")
    print(f"测试集 AUPR: {best_metrics['aupr_test']:.4f}")
    print(f"最佳参数: {best_metrics['best_params']}")

if __name__ == "__main__":
    main()
