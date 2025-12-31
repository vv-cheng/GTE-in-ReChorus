import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import argparse
import os
import random
import sys
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_gte_data(data_name):
    data_dir = f"./{data_name}"
    
    if not os.path.exists(data_dir):
        print(f"错误：数据目录不存在 {os.path.abspath(data_dir)}")
        return None, None
    
    train_path = os.path.join(data_dir, 'train_mat.pkl')
    test_path = os.path.join(data_dir, 'test_mat.pkl')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        return None, None
    
    try:
        with open(train_path, 'rb') as f:
            train_mat = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_mat = pickle.load(f)
        return train_mat, test_mat
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return None, None

def coo_to_interactions(matrix):
    interactions = []
    if matrix is not None and hasattr(matrix, 'row') and hasattr(matrix, 'col'):
        for i, j in zip(matrix.row, matrix.col):
            interactions.append((i, j))
    return interactions

def generate_rechorus_split(train_interactions, test_interactions, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    train_interactions = [(u+1, i+1) for u, i in train_interactions]
    test_interactions = [(u+1, i+1) for u, i in test_interactions]
    
    user_interactions = defaultdict(list)
    for user, item in train_interactions + test_interactions:
        user_interactions[user].append(item)
    
    user_timestamps = {}
    for user in user_interactions:
        num_interactions = len(user_interactions[user])
        timestamps = sorted(random.sample(range(1, 1000000), num_interactions))
        user_timestamps[user] = timestamps
    
    train_data = []
    dev_data = []
    test_data = []
    
    for user, items in user_interactions.items():
        timestamps = user_timestamps[user]
        
        if len(items) >= 3:
            train_data.append((user, items[0], timestamps[0]))
            for i in range(1, len(items)-2):
                train_data.append((user, items[i], timestamps[i]))
            dev_data.append((user, items[-2], timestamps[-2]))
            test_data.append((user, items[-1], timestamps[-1]))
        elif len(items) == 2:
            train_data.append((user, items[0], timestamps[0]))
            test_data.append((user, items[1], timestamps[1]))
        else:
            train_data.append((user, items[0], timestamps[0]))
    
    return train_data, dev_data, test_data

def prepare_negative_samples_with_check(train_data, dev_data, test_data, num_negatives=99, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    user_history = defaultdict(set)
    all_observed_items = set()
    
    for user, item, _ in train_data + dev_data + test_data:
        user_history[user].add(item)
        all_observed_items.add(item)
    
    all_items_list = list(all_observed_items)
    max_item_id = max(all_observed_items) if all_observed_items else 0
    
    def generate_negatives_for_user(user, pos_item, user_hist):
        candidate_negs = [item for item in all_items_list 
                         if item not in user_hist and item != pos_item]
        
        if not candidate_negs:
            candidate_negs = [item for item in all_items_list if item != pos_item]
        
        if len(candidate_negs) >= num_negatives:
            negs = random.sample(candidate_negs, num_negatives)
        else:
            negs = list(np.random.choice(candidate_negs, num_negatives, replace=True))
        
        assert all(neg <= max_item_id for neg in negs), f"负样本ID超出范围: {negs}"
        
        return negs
    
    dev_with_negs = []
    for user, item, time in dev_data:
        user_hist = user_history[user]
        negs = generate_negatives_for_user(user, item, user_hist)
        dev_with_negs.append((user, item, time, negs))
    
    test_with_negs = []
    for user, item, time in test_data:
        user_hist = user_history[user]
        negs = generate_negatives_for_user(user, item, user_hist)
        test_with_negs.append((user, item, time, negs))
    
    return dev_with_negs, test_with_negs, len(all_items_list), max_item_id

def save_rechorus_files(data_name, train_data, dev_data, test_data, output_dir):
    dataset_dir = os.path.join(output_dir, data_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    train_df = pd.DataFrame(train_data, columns=['user_id', 'item_id', 'time'])
    train_df = train_df.sort_values(by=['user_id', 'time']).reset_index(drop=True)
    train_file = os.path.join(dataset_dir, 'train.csv')
    train_df.to_csv(train_file, sep='\t', index=False)
    
    dev_df = pd.DataFrame(dev_data, columns=['user_id', 'item_id', 'time', 'neg_items'])
    dev_df['neg_items'] = dev_df['neg_items'].apply(lambda x: str(list(x)) if isinstance(x, list) else str(x))
    dev_file = os.path.join(dataset_dir, 'dev.csv')
    dev_df.to_csv(dev_file, sep='\t', index=False)
    
    test_df = pd.DataFrame(test_data, columns=['user_id', 'item_id', 'time', 'neg_items'])
    test_df['neg_items'] = test_df['neg_items'].apply(lambda x: str(list(x)) if isinstance(x, list) else str(x))
    test_file = os.path.join(dataset_dir, 'test.csv')
    test_df.to_csv(test_file, sep='\t', index=False)
    
    return train_df, dev_df, test_df, dataset_dir

def verify_dataset_compatibility(data_name, dataset_dir):
    try:
        train_df = pd.read_csv(os.path.join(dataset_dir, 'train.csv'), sep='\t')
        dev_df = pd.read_csv(os.path.join(dataset_dir, 'dev.csv'), sep='\t')
        test_df = pd.read_csv(os.path.join(dataset_dir, 'test.csv'), sep='\t')
        
        min_user_id = min(train_df['user_id'].min(), dev_df['user_id'].min(), test_df['user_id'].min())
        min_item_id = min(train_df['item_id'].min(), dev_df['item_id'].min(), test_df['item_id'].min())
        
        if min_user_id == 0 or min_item_id == 0:
            return False
        
        assert train_df.shape[1] == 3
        assert dev_df.shape[1] == 4
        assert test_df.shape[1] == 4
        
        expected_train_cols = ['user_id', 'item_id', 'time']
        expected_dev_test_cols = ['user_id', 'item_id', 'time', 'neg_items']
        
        assert list(train_df.columns) == expected_train_cols
        assert list(dev_df.columns) == expected_dev_test_cols
        assert list(test_df.columns) == expected_dev_test_cols
        
        try:
            dev_df['neg_items'] = dev_df['neg_items'].apply(lambda x: eval(str(x)) if pd.notna(x) else [])
            test_df['neg_items'] = test_df['neg_items'].apply(lambda x: eval(str(x)) if pd.notna(x) else [])
        except:
            return False
        
        dev_neg_counts = dev_df['neg_items'].apply(len)
        test_neg_counts = test_df['neg_items'].apply(len)
        
        if len(dev_df) > 0:
            assert dev_neg_counts.nunique() == 1
        if len(test_df) > 0:
            assert test_neg_counts.nunique() == 1
        
        return True
        
    except:
        return False

def convert_gte_to_rechorus(data_name, num_negatives=99, output_dir="../"):
    print(f"转换数据集: {data_name}")
    
    try:
        train_mat, test_mat = load_gte_data(data_name)
        
        if train_mat is None or test_mat is None:
            print(f"✗ 无法加载数据，跳过 {data_name}")
            return False
        
        train_interactions = coo_to_interactions(train_mat)
        test_interactions = coo_to_interactions(test_mat)
        
        train_data, dev_data, test_data = generate_rechorus_split(train_interactions, test_interactions)
        
        dev_with_negs, test_with_negs, actual_num_items, max_item_id = prepare_negative_samples_with_check(
            train_data, dev_data, test_data, num_negatives
        )
        
        train_df, dev_df, test_df, dataset_dir = save_rechorus_files(
            data_name, train_data, dev_with_negs, test_with_negs, output_dir
        )
        
        if verify_dataset_compatibility(data_name, dataset_dir):
            print(f"✓ {data_name} 转换成功")
        else:
            print(f"⚠ {data_name} 转换完成但验证失败")
        
        return True
        
    except Exception as e:
        print(f"✗ 转换失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='将GTE数据集转换为rechorus格式')
    parser.add_argument('--data', default='amazon_beauty', type=str, help='数据集名称')
    parser.add_argument('--negatives', default=99, type=int, help='每个正样本的负样本数量')
    parser.add_argument('--output_dir', default='../', type=str, help='输出目录')
    parser.add_argument('--all', action='store_true', help='转换所有数据集')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.all:
        datasets = ['amazon_beauty', 'sparse_tmall', 'douban', 'gowalla', 'tmall', 'yelp']
        results = {}
        
        for dataset in datasets:
            success = convert_gte_to_rechorus(dataset, args.negatives, args.output_dir)
            results[dataset] = "成功" if success else "失败"
        
        print("转换总结:")
        for dataset, status in results.items():
            print(f"  {dataset}: {status}")
        
    else:
        convert_gte_to_rechorus(args.data, args.negatives, args.output_dir)

if __name__ == "__main__":
    main()