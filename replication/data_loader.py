import os

def load_data(filenames):
    """Load training and testing data from specified filenames."""
    X, Y = [], []
    for filename in filenames:
        x_fold, y_fold = [], []
        with open(filename, 'r') as f:
            x_inst, y_inst = [], []
            for line in f:
                parts = line.strip().split()
                if not parts:
                    if x_inst or y_inst:
                        x_fold.append(x_inst)
                        y_fold.append(y_inst)
                        x_inst, y_inst = [], []
                    continue
                
                label = parts[0]
                features = {feat.split(':')[0]: float(feat.split(':')[1]) for feat in parts[1:]}
                x_inst.append(features)
                y_inst.append(label)
                
            if x_inst or y_inst:
                x_fold.append(x_inst)
                y_fold.append(y_inst)
                
        X.extend(x_fold)
        Y.extend(y_fold)
    return X, Y