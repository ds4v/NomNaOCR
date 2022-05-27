import tensorflow as tf
from tensorflow.keras.models import clone_model
from sklearn.model_selection import KFold


def kfold_decorator(img_paths, labels, n_splits, random_state=None, is_subclassed_model=False):
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    valid_datasets = []
    best_epochs = []
    histories = []
    models = []

    def decorator(func):
        def wrapper(model, *args, **kwargs):
            for fold_idx, (train_idxs, valid_idxs) in enumerate(kf.split(img_paths, labels)):
                if not is_subclassed_model: reset_model = clone_model(model) 
                else: reset_model = model.__class__.from_config(model.get_config())
                reset_model._name = f'Model_{fold_idx + 1:02d}'
                print(f'============== Fold {fold_idx + 1:02d} training ==============')
                valid_tf_dataset, best_epoch, history, reset_model = func(reset_model, train_idxs, valid_idxs)
                valid_datasets.append((valid_tf_dataset, valid_idxs))
                best_epochs.append(best_epoch)
                histories.append(history)
                models.append(reset_model)
            return valid_datasets, best_epochs, histories, models
        return wrapper
    return decorator


def get_best_fold(valid_datasets, best_epochs, histories, models, verbose=1):
    best_loss = float('inf')
    for fold_idx, model in enumerate(models):
        steps = None
        valid_tf_dataset, valid_idxs = valid_datasets[fold_idx]
        dataset_size = tf.data.experimental.cardinality(valid_tf_dataset).numpy()

        if dataset_size == tf.data.experimental.INFINITE_CARDINALITY:
            for batch in valid_tf_dataset.take(1): 
                batch_size = batch['image'].shape[0]
            steps = len(valid_idxs) // batch_size

        result = model.evaluate(valid_tf_dataset, steps=steps, verbose=verbose, return_dict=True)
        if result['loss'] < best_loss:
            best_loss = result['loss']
            best_fold_idx = fold_idx
            
    return (
        valid_datasets[best_fold_idx], 
        best_epochs[best_fold_idx],
        histories[best_fold_idx],
        models[best_fold_idx],
        best_fold_idx, 
        best_loss
    )
