import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import time
from datetime import datetime
import warnings
import itertools
from collections import defaultdict

warnings.filterwarnings("ignore")

# 检查 GPU 是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class CNNModel(nn.Module):
    def __init__(self, input_dim, out_channels1=32, out_channels2=64, 
                 kernel_size1=3, kernel_size2=3, pool_size1=2, pool_size2=2,
                 hidden_dim=128, dropout_rate=0.5):
        super(CNNModel, self).__init__()
        
        # 计算卷积后的维度
        def calc_conv_output_size(input_size, kernel_size, padding=1, stride=1):
            return (input_size + 2 * padding - kernel_size) // stride + 1
        
        def calc_pool_output_size(input_size, pool_size, stride=None):
            if stride is None:
                stride = pool_size
            return (input_size - pool_size) // stride + 1
        
        # 第一层卷积
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=out_channels1, 
                              kernel_size=kernel_size1, padding=1)
        conv1_out = calc_conv_output_size(input_dim, kernel_size1)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_size1)
        pool1_out = calc_pool_output_size(conv1_out, pool_size1)
        
        # 第二层卷积
        self.conv2 = nn.Conv1d(in_channels=out_channels1, out_channels=out_channels2, 
                              kernel_size=kernel_size2, padding=1)
        conv2_out = calc_conv_output_size(pool1_out, kernel_size2)
        self.pool2 = nn.MaxPool1d(kernel_size=pool_size2)
        pool2_out = calc_pool_output_size(conv2_out, pool_size2)
        
        # 计算全连接层输入维度
        fc_input_dim = out_channels2 * pool2_out
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(fc_input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
        # 保存维度信息用于调试
        self.fc_input_dim = fc_input_dim

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

def train_cnn_model(params, xtrain, ytrain, xval, yval, input_dim, num_epochs=100, patience=10):
    """训练单个CNN模型"""
    # 准备数据
    xtrain_tensor = torch.tensor(xtrain, dtype=torch.float32).unsqueeze(1)
    ytrain_tensor = torch.tensor(ytrain, dtype=torch.float32)
    xval_tensor = torch.tensor(xval, dtype=torch.float32).unsqueeze(1)
    yval_tensor = torch.tensor(yval, dtype=torch.float32)
    
    train_dataset = TensorDataset(xtrain_tensor, ytrain_tensor)
    val_dataset = TensorDataset(xval_tensor, yval_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    model = CNNModel(input_dim=input_dim, **params).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            # 保存最佳模型
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    return model, best_val_loss, best_epoch

def evaluate_model(model, x, y):
    """评估模型性能"""
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model.eval()
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend((outputs > 0.5).cpu().numpy())
            y_pred_proba.extend(outputs.cpu().numpy())
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auroc': roc_auc_score(y_true, y_pred_proba),
        'aupr': average_precision_score(y_true, y_pred_proba)
    }
    
    return metrics, y_true, y_pred, y_pred_proba

def run_cnn_experiment(param_combination, random_state, input_dim):
    """运行单次CNN实验"""
    print(f"\n开始实验，随机种子: {random_state}")
    print(f"参数组合: {param_combination}")
    start_time = time.time()
    
    # 加载数据
    data = pd.read_csv("/home/cyyue/Data/DTI_light/feature_con/tensor_HKG_MACCS.csv")
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # 数据标准化
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    # 划分训练验证测试集
    x_temp, xtest, y_temp, ytest = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )
    xtrain, xval, ytrain, yval = train_test_split(
        x_temp, y_temp, test_size=0.25, random_state=random_state, stratify=y_temp
    )
    
    # 训练模型
    model, best_val_loss, best_epoch = train_cnn_model(
        param_combination, xtrain, ytrain, xval, yval, input_dim
    )
    
    # 评估测试集
    test_metrics, y_true, y_pred, y_pred_proba = evaluate_model(model, xtest, ytest)
    
    
    results = {
        'random_state': random_state,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        **test_metrics,
        'params': str(param_combination)
    }
    
    return results, model, (y_true, y_pred, y_pred_proba)

def main():
    # 定义参数搜索空间
    param_grid = {
        'out_channels1': [16，32],
        'out_channels2': [64，128],
        'kernel_size1': [3, 5],
        'kernel_size2': [3, 5],
        'pool_size1': [2, 3],
        'pool_size2': [2, 3],
        'hidden_dim': [64，128, 256],
        'dropout_rate': [0.3, 0.4, 0.5]
    }
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_combinations = list(itertools.product(*param_grid.values()))
    
    # 
    selected_combinations = param_combinations
    
    input_dim = 422  # 输入特征维度
    
    all_results = []
    best_models = []
    all_predictions = []
    
    print(f"=== 开始CNN参数搜索 ===")
    print(f"总参数组合数: {len(param_combinations)}")
    print(f"测试组合数: {len(selected_combinations)}")
    
    # 对每个参数组合运行多次实验
    for i, param_values in enumerate(selected_combinations):
        param_dict = dict(zip(param_names, param_values))
        print(f"\n=== 测试参数组合 {i+1}/{len(selected_combinations)} ===")
        print(f"参数: {param_dict}")
        
        # 对每个参数组合运行3次实验
        for run in range(3):
            random_state = 42 + i * 10 + run
            try:
                results, model, predictions = run_cnn_experiment(
                    param_dict, random_state, input_dim
                )
                all_results.append(results)
                best_models.append(model)
                all_predictions.append(predictions)
                
                print(f"运行 {run+1}: AUROC={results['auroc']:.4f}, "
                    f"AUPR={results['aupr']:.4f}, F1={results['f1']:.4f}, "
                    f"Accuracy={results['accuracy']:.4f}")
                
            except Exception as e:
                print(f"参数组合 {i+1} 运行 {run+1} 失败: {e}")
                continue

    if not all_results:
        print("所有实验都失败了！")
        return

    # 分析结果
    results_df = pd.DataFrame(all_results)

    # 按参数分组统计所有指标
    param_results = results_df.groupby('params').agg({
        'auroc': ['mean', 'std', 'count'],
        'aupr': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'best_epoch': ['mean', 'std'],
        'best_val_loss': ['mean', 'std']
    }).round(4)

    print("\n" + "="*80)
    print("=== CNN参数搜索结果汇总 ===")
    print(f"总实验次数: {len(all_results)}")
    print(f"成功实验次数: {len(all_results)}")

    print("\n各参数组合性能排名 (按AUROC):")
    sorted_results = param_results.sort_values(('auroc', 'mean'), ascending=False)

    for i, (params, row) in enumerate(sorted_results.iterrows()):
        print(f"\n{i+1}. 参数组合: {params}")
        print(f"   实验次数: {row[('auroc', 'count')]}")
        print(f"   AUROC:    {row[('auroc', 'mean')]:.4f} ± {row[('auroc', 'std')]:.4f}")
        print(f"   AUPR:     {row[('aupr', 'mean')]:.4f} ± {row[('aupr', 'std')]:.4f}")
        print(f"   F1:       {row[('f1', 'mean')]:.4f} ± {row[('f1', 'std')]:.4f}")
        print(f"   Accuracy: {row[('accuracy', 'mean')]:.4f} ± {row[('accuracy', 'std')]:.4f}")
        print(f"   Precision:{row[('precision', 'mean')]:.4f} ± {row[('precision', 'std')]:.4f}")
        print(f"   Recall:   {row[('recall', 'mean')]:.4f} ± {row[('recall', 'std')]:.4f}")
        print(f"   最佳轮次: {row[('best_epoch', 'mean')]:.0f} ± {row[('best_epoch', 'std')]:.1f}")
        print(f"   验证损失: {row[('best_val_loss', 'mean')]:.4f} ± {row[('best_val_loss', 'std')]:.4f}")

    # 保存详细结果
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists('model'):
        os.makedirs('model')

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'results/cnn_maccs_hkg_results_{timestamp}.csv', index=False)
    param_results.to_csv(f'results/cnn_maccs_hkg_summary_{timestamp}.csv')

    # 选择最佳参数组合
    best_params_str = sorted_results.index[0]
    best_row = sorted_results.iloc[0]

    print("\n" + "="*80)
    print("=== 最佳参数组合 ===")
    print(f"参数: {best_params_str}")
    print(f"AUROC:    {best_row[('auroc', 'mean')]:.4f} ± {best_row[('auroc', 'std')]:.4f}")
    print(f"AUPR:     {best_row[('aupr', 'mean')]:.4f} ± {best_row[('aupr', 'std')]:.4f}")
    print(f"F1:       {best_row[('f1', 'mean')]:.4f} ± {best_row[('f1', 'std')]:.4f}")
    print(f"Accuracy: {best_row[('accuracy', 'mean')]:.4f} ± {best_row[('accuracy', 'std')]:.4f}")
    print(f"Precision:{best_row[('precision', 'mean')]:.4f} ± {best_row[('precision', 'std')]:.4f}")
    print(f"Recall:   {best_row[('recall', 'mean')]:.4f} ± {best_row[('recall', 'std')]:.4f}")

    # 保存最佳模型的详细预测结果
    best_idx = results_df[results_df['params'] == best_params_str]['auroc'].idxmax()
    best_model = best_models[best_idx]
    best_predictions = all_predictions[best_idx]

    y_true, y_pred, y_pred_proba = best_predictions
    best_test_results = pd.DataFrame({
        'True_Label': y_true,
        'Predicted_Label': y_pred,
        'Predicted_Probability': y_pred_proba
    })
    best_test_results.to_csv(f"model/CNN_MACCS_HKG_best_predictions_{timestamp}.csv", index=False)

    # 保存最佳模型
    torch.save(best_model.state_dict(), f'model/CNN_MACCS_HKG_best_{timestamp}.pth')

    print(f"\n最佳模型和预测结果已保存，时间戳: {timestamp}")

if __name__ == "__main__":
    main()

