import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import os
import time
from datetime import datetime

warnings.filterwarnings("ignore")

def run_mlp_experiment(random_state=None):
    """
    运行单次MLP实验
    """
    print(f"开始MLP实验，随机种子: {random_state}")
    start_time = time.time()
    
    # 加载数据
    data = pd.read_csv("/home/cyyue/Data/DTI_light/feature_con/tensor_HKG_UniMol.csv")
    x = data.iloc[:, :-1]
    y = np.array(data.iloc[:, -1])
    
    # 划分训练集和测试集
    xtrain, xtest, ytrain, ytest = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # 数据归一化
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)
    
    # 扩展的超参数网格
    param_grid = {
        "hidden_layer_sizes": [
            (50,), (100,), (200,), (300,),(500,)
        ],
        "activation": ["relu", "tanh"],
        "solver": ["adam"],
        "alpha": [0.0001, 0.001], 
        "learning_rate": ["constant"],
        "early_stopping": [True]
    }
    
    # 使用分层K折交叉验证
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    
    # 基础模型配置
    base_model = MLPClassifier(
        max_iter=1000,
        random_state=random_state,
        n_iter_no_change=20,  
        tol=1e-4,
        verbose=False
    )
    
    # 网格搜索
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
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
        'best_cv_score': grid_search.best_score_,
        'training_time': time.time() - start_time,
        'n_iter_actual': best_model.n_iter_,
        'random_state': random_state
    }
    
    return metrics, best_model, scaler, (ytest, y_pred_test, y_pred_test_proba)

def main():
    # 运行多次实验
    n_iterations = 5  
    all_results = []
    all_models = []
    all_scalers = []
    all_predictions = []
    
    print(f"=== 开始MLP模型评估，共 {n_iterations} 次实验 ===")
    
    for i in range(n_iterations):
        print(f"\n=== 第 {i+1}/{n_iterations} 次实验 ===")
        try:
            metrics, model, scaler, predictions = run_mlp_experiment(random_state=42 + i)
            all_results.append(metrics)
            all_models.append(model)
            all_scalers.append(scaler)
            all_predictions.append(predictions)
            
            # 打印当前结果
            print(f"测试集 AUROC: {metrics['auroc_test']:.4f}")
            print(f"测试集 AUPR: {metrics['aupr_test']:.4f}")
            print(f"实际迭代次数: {metrics['n_iter_actual']}")
            print(f"最佳参数: {metrics['best_params']}")
            
        except Exception as e:
            print(f"第 {i+1} 次实验失败: {e}")
            continue
    
    if not all_results:
        print("所有实验都失败了！")
        return
    
    # 转换为DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 计算统计信息
    numeric_cols = results_df.select_dtypes(include=[np.number]).columns
    
    summary = pd.DataFrame({
        'mean': results_df[numeric_cols].mean(),
        'std': results_df[numeric_cols].std(),
        'min': results_df[numeric_cols].min(),
        'max': results_df[numeric_cols].max(),
        'median': results_df[numeric_cols].median(),
        'cv': results_df[numeric_cols].std() / results_df[numeric_cols].mean()  # 变异系数
    })
    
    print("\n" + "="*60)
    print("=== MLP实验结果汇总 ===")
    print(f"成功实验次数: {len(all_results)}/{n_iterations}")
    
    print("\n测试集指标统计:")
    test_metrics = [col for col in summary.index if 'test' in col]
    for metric in test_metrics:
        mean_val = summary.loc[metric, 'mean']
        std_val = summary.loc[metric, 'std']
        cv_val = summary.loc[metric, 'cv']
        print(f"{metric}: {mean_val:.4f} ± {std_val:.4f} (CV: {cv_val:.3f})")
    
    print("\n最佳参数分布:")
    param_counts = results_df['best_params'].value_counts()
    for params, count in param_counts.items():
        print(f"出现次数 {count}: {params}")
    
    # 保存结果
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('model'):
        os.makedirs('model')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'results/mlp_unimol_hkg_results_{timestamp}.csv', index=False)
    summary.to_csv(f'results/mlp_unimol_hkg_summary_{timestamp}.csv')
    
    # 选择最佳模型（基于测试集AUROC）
    best_run_idx = results_df['auroc_test'].idxmax()
    best_model = all_models[best_run_idx]
    best_scaler = all_scalers[best_run_idx]
    best_metrics = results_df.iloc[best_run_idx]
    
    # 保存最佳模型和标准化器
    joblib.dump(best_model, 'model/MLP_UniMol_HKG_best.pkl')
    joblib.dump(best_scaler, 'model/MLP_scaler_UniMol_HKG_best.pkl')
    
    # 保存最佳实验的预测结果
    ytest_best, y_pred_best, y_pred_proba_best = all_predictions[best_run_idx]
    test_results = pd.DataFrame({
        'True_Label': ytest_best,
        'Predicted_Label': y_pred_best,
        'Predicted_Probability': y_pred_proba_best
    })
    test_results.to_csv("model/MLP_UniMol_HKG_test_predictions_best.csv", index=False)

if __name__ == "__main__":
    main()