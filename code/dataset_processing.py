import pandas as pd
import numpy as np
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem, SaltRemover, MACCSkeys
from rdkit import RDLogger
from transformers import BertModel, BertTokenizer
from sklearn.decomposition import PCA
from unimol_tools import UniMolRepr
import warnings
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

class DTIFeatureProcessor:
    def __init__(self, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def standardize_smiles(self, input_file, output_file, smiles_column='Compound SMILES'):
        """
        标准化SMILES：去盐、中和电荷、规范化
        """
        print("开始标准化SMILES...")
        data = pd.read_csv(input_file, sep="\t")
        smiles_list = data[smiles_column]
        
        salt_remover = SaltRemover.SaltRemover()
        
        def neutralize_charges(mol):
            for charge in [1, -1, 2, -2, 3, -3]:
                pattern = Chem.MolFromSmarts(f"[{charge:+d}]")
                while mol.HasSubstructMatch(pattern):
                    at = mol.GetSubstructMatch(pattern)[0]
                    mol.GetAtomWithIdx(at).SetFormalCharge(0)
            return mol
        
        canonical_smiles_list = []
        valid_indices = []
        
        for index, smiles in enumerate(smiles_list):
            try:
                molecule = Chem.MolFromSmiles(smiles)
                if molecule:
                    molecule = salt_remover.StripMol(molecule)
                    molecule = neutralize_charges(molecule)
                    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
                    canonical_smiles_list.append(canonical_smiles)
                    valid_indices.append(index)
            except Exception as e:
                continue
        
        updated_data = data.loc[valid_indices].copy()
        updated_data['Canonical_SMILES_v2'] = canonical_smiles_list
        
        print(f"原始数据 {len(smiles_list)} 个分子，标准化后剩余 {len(updated_data)} 个分子")
        updated_data.to_csv(output_file, sep="\t", index=False)
        return updated_data
    
    def generate_fingerprints(self, input_file, output_prefix, smiles_column='Canonical_SMILES_v2', 
                            index_column='Lindexid', fingerprint_types=['morgan']):
        """
        生成分子指纹特征
        """
        print("开始生成分子指纹...")
        data = pd.read_csv(input_file, sep="\t")
        smiles_list = data[smiles_column]
        smiles_to_index = data.set_index(smiles_column)[index_column].to_dict()
        
        fingerprint_functions = {
            'morgan': lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512),
            'atom_pair': lambda mol: AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=512),
            'rdkit': lambda mol: Chem.RDKFingerprint(mol, fpSize=512),
            'topological': lambda mol: Chem.RDKFingerprint(mol, fpSize=512),
            'maccs': lambda mol: MACCSkeys.GenMACCSKeys(mol)
        }
        
        for fp_type in fingerprint_types:
            if fp_type not in fingerprint_functions:
                print(f"不支持的指纹类型: {fp_type}")
                continue
                
            fingerprints = []
            fp_func = fingerprint_functions[fp_type]
            n_bits = 166 if fp_type == 'maccs' else 512
            
            for smiles in smiles_list:
                try:
                    molecule = Chem.MolFromSmiles(smiles)
                    if molecule:
                        index = smiles_to_index[smiles]
                        fp = fp_func(molecule)
                        fp_list = list(fp)
                        fingerprints.append([index] + fp_list)
                except Exception as e:
                    print(f"处理SMILES出错: {smiles}, 错误: {e}")
            
            columns = ["Lindex"] + [f"{fp_type.capitalize()}Feature_{i}" for i in range(n_bits)]
            fp_df = pd.DataFrame(fingerprints, columns=columns)
            output_file = f"{output_prefix}_{fp_type}_{n_bits}.csv"
            fp_df.to_csv(output_file, index=False)
            print(f"{fp_type}指纹已保存到 {output_file}")
    
    def generate_unimol_features(self, input_file, output_file, smiles_column='SMILES', 
                               index_column='source', model_name='unimolv1'):
        """
        生成UniMol分子表征
        """
        print("开始生成UniMol特征...")
        data = pd.read_csv(input_file, sep="\t")
        smiles_list = data[smiles_column]
        
        clf = UniMolRepr(data_type='molecule', remove_hs=False, model_name=model_name)
        
        def get_unimol_repr(smiles_list):
            unimol_repr_list = []
            error_smiles_list = []
            
            for i, smiles in enumerate(smiles_list, 1):
                try:
                    unimol_repr = clf.get_repr([smiles], return_atomic_reprs=True)
                    unimol_repr_list.append(unimol_repr['cls_repr'][0])
                    if i % 100 == 0:
                        print(f"已处理 {i} 个分子")
                except Exception as e:
                    print(f"处理SMILES出错: {smiles}, 错误: {e}")
                    unimol_repr_list.append(np.zeros(512))
                    error_smiles_list.append(smiles)
            
            return np.array(unimol_repr_list), error_smiles_list
        
        unimol_repr_list, error_smiles_list = get_unimol_repr(smiles_list)
        
        data['UniMol_repr'] = unimol_repr_list.tolist()
        unimol_features = np.vstack(data['UniMol_repr'])
        
        unimol_columns = [index_column] + [f"UniMolFeature_{i}" for i in range(512)]
        unimol_df = pd.DataFrame(np.hstack([data[[index_column]].values, unimol_features]), 
                               columns=unimol_columns)
        
        unimol_df.to_csv(output_file, index=False)
        print(f"UniMol特征已保存到 {output_file}")
        
        if error_smiles_list:
            print(f"有 {len(error_smiles_list)} 个SMILES处理出错")
            with open('error_smiles.txt', 'w') as f:
                for smiles in error_smiles_list:
                    f.write(smiles + '\n')
    
    def generate_protein_bert_features(self, input_file, output_file, seq_column='seq', 
                                    index_column='index', model_path="/home/cyyue/Install/proteinbert"):
        """
        生成ProteinBert蛋白表征，并将特征向量展开为多列格式
        """
        print("开始生成ProteinBert特征...")
        df = pd.read_csv(input_file, sep="\t")
        
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        model = BertModel.from_pretrained(model_path)
        
        def protein_to_embedding(protein_sequence):
            inputs = tokenizer(protein_sequence, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs)
            return outputs.last_hidden_state.mean(1).squeeze().detach().numpy()
        
        embeddings_list = []
        for _, row in df.iterrows():
            sequence = row[seq_column]
            protein_index = row[index_column]
            embedding = protein_to_embedding(sequence)
            embeddings_list.append([protein_index] + embedding.tolist())
        

        columns = [index_column] + [f'Pbert_{i}' for i in range(1024)]
        tensor_df = pd.DataFrame(embeddings_list, columns=columns)
        
        # 应用PCA降维到128维
        embeddings_array = tensor_df.iloc[:, 1:].values  
        
        pca = PCA(n_components=128)
        reduced_embeddings = pca.fit_transform(embeddings_array)
        
        
        reduced_columns = [index_column] + [f'Pbert_{i}' for i in range(128)]
        reduced_df = pd.DataFrame(
            np.column_stack([tensor_df[index_column].values, reduced_embeddings]),
            columns=reduced_columns
        )
        
        # 设置索引列为正确的数据类型（如果是数字索引）
        if reduced_df[index_column].dtype == 'object':
            try:
                reduced_df[index_column] = reduced_df[index_column].astype(int)
            except:
                pass  # 如果无法转换为整数，保持原样
        
        reduced_df.to_csv(output_file, index=False, sep='\t', float_format='%.6e')
        print(f"ProteinBert特征已保存到 {output_file}")
    
    def generate_protein_kg_features(self, node_index_file, kg_file, output_file, 
                                  embedding_dim=128, walks_per_node=10, walk_length=20):
        """
        生成知识图谱蛋白节点特征
        """
        print("开始生成知识图谱蛋白特征...")
        node_index = pd.read_csv(node_index_file, sep="\t")
        node_names = node_index['node_name'].values
        
        data = pd.read_csv(kg_file)
        edge_index = torch.tensor(data[['head', 'tail']].values.T, dtype=torch.long).to(self.device)
        
        num_nodes = edge_index.max().item() + 1
        node_features = torch.ones((num_nodes, 64)).to(self.device)
        
        graph_data = Data(x=node_features, edge_index=edge_index)
        
        model = Node2Vec(
            graph_data.edge_index,
            embedding_dim=embedding_dim,
            walks_per_node=walks_per_node,
            walk_length=walk_length,
            context_size=10,
            p=1.0,
            q=1.0,
            num_negative_samples=1
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loader = model.loader(batch_size=128, shuffle=True)
        
        def train():
            model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader)
        
        for epoch in range(10):
            loss = train()
            print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
        
        z = model().cpu().detach().numpy()
        columns = ['node_name'] + [f'PKG_{i}' for i in range(embedding_dim)]
        output_df = pd.DataFrame(data=[node_names, *z.T]).T
        output_df.columns = columns
        
        filtered_result_df = output_df[output_df['node_name'].isin(node_index['node_name'])]
        filtered_result_df.to_csv(output_file, index=False)
        print(f"知识图谱蛋白特征已保存到 {output_file}")
    
    def combine_features(self, protein_tensor_file, protein_kg_file, ligand_features_file, 
                       train_data_file, output_file):
        """
        组合所有特征并创建训练数据集
        """
        print("开始组合特征...")
        protein_tensor = pd.read_csv(protein_tensor_file)
        protein_kg = pd.read_csv(protein_kg_file)
        ligand_features = pd.read_csv(ligand_features_file)
        
        protein_tensor.set_index(protein_tensor.columns[0], inplace=True)
        protein_kg.set_index(protein_kg.columns[0], inplace=True)
        ligand_features.set_index(ligand_features.columns[0], inplace=True)
        
        train_data_df = pd.read_csv(train_data_file, sep="\t")
        Pindex_list = train_data_df["Pindex"]
        Lindex_list = train_data_df["Lindex"]
        labels = train_data_df["Label"]
        
        train_data = []
        missing_count = 0
        
        for i, (Pindex, Lindex, label) in enumerate(zip(Pindex_list, Lindex_list, labels), 1):
            if (Pindex not in protein_tensor.index or 
                Pindex not in protein_kg.index or 
                Lindex not in ligand_features.index):
                missing_count += 1
                continue
            
            protein_tensor_vector = protein_tensor.loc[Pindex].values
            protein_kg_vector = protein_kg.loc[Pindex].values
            ligand_vector = ligand_features.loc[Lindex].values
            
            combined_vector = (list(protein_tensor_vector) + 
                             list(protein_kg_vector) + 
                             list(ligand_vector) + 
                             [label])
            train_data.append(combined_vector)
            
            if i % 1000 == 0:
                print(f"已处理 {i} 个样本")
        
        header = (list(protein_tensor.columns) + 
                 list(protein_kg.columns) + 
                 list(ligand_features.columns) + 
                 ["Label"])
        
        final_train_data_df = pd.DataFrame(train_data, columns=header)
        final_train_data_df.to_csv(output_file, index=False)
        
        print(f"特征组合完成，共 {len(train_data)} 个样本，缺失 {missing_count} 个样本")
        print(f"最终数据集已保存到 {output_file}")

# 使用示例
if __name__ == "__main__":
    processor = DTIFeatureProcessor()
    
    # 1. 标准化SMILES
    processor.standardize_smiles('ligand.txt', 'ligand_standardized.txt')
    
    # 2. 生成分子指纹
    processor.generate_fingerprints('ligand_standardized.txt', 'ligand_fp', 
                                  fingerprint_types=['morgan', 'maccs'])
    
    # 3. 生成UniMol特征
    processor.generate_unimol_features('ligand_standardized.txt', 'ligand_unimol_512.csv')
    
    # 4. 生成ProteinBert特征
    processor.generate_protein_bert_features('seq.txt', 'tensor.csv')
    
    # 5. 生成知识图谱特征
    processor.generate_protein_kg_features('node_index.txt', 'kg_index.csv', 
                                         'node_feature.csv')
    
    # 6. 组合特征
    processor.combine_features(
        'tensor.csv',
        'node_feature.csv',
        'ligand_fp_Morgan_2_512.csv',
        'train_data.txt',
        'train_tensor_KG_Morgan.csv'
    )
