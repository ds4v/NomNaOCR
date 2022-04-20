import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties


def draw_predicted_text(label, pred_label, fontdict, text_x):
    label = label.replace('[UNK]', '?')
    label_length, pred_length = len(label), len(pred_label)

    if pred_label == label:
        fontdict['color'] = 'green'
        plt.text(text_x, 0, '\n'.join(pred_label), fontdict=fontdict)
        return 

    pred_start, start, end = 0, 0, 0
    while start <= end < label_length:
        text_y = end * label_length * 5
        actual_char = '[UNK]' if label[end] == '?' else label[end]

        if label[start:end + 1] in pred_label[pred_start:pred_length]:
            fontdict['color'] = 'dodgerblue'
            plt.text(text_x, text_y, actual_char, fontdict=fontdict)
        else:
            if end < pred_length and end + 1 < label_length and \
                pred_label[end] == label[end + 1]:
                fontdict['color'] = 'gray'
                plt.text(text_x, text_y, actual_char, fontdict=fontdict)
            elif end < pred_length:
                fontdict['color'] = 'red'
                plt.text(text_x, text_y, pred_label[end], fontdict=fontdict)
                fontdict['color'] = 'black'
                plt.text(text_x + 35, text_y, actual_char, fontdict=fontdict)
            else: 
                fontdict['color'] = 'gray'
                plt.text(text_x, text_y, actual_char, fontdict=fontdict)
                
            pred_start = end
            start = end + 1
        end += 1


def visualize_images_labels(
    img_paths, 
    labels, # shape == (batch_size, max_length)
    pred_labels = None, # shape == (batch_size, max_length)
    figsize = (15, 8),
    subplot_size = (2, 8), # tuple: (rows, columns) to display
    legend_loc = None, # Only for predictions,
    annotate_loc = None, # Only for predictions
    font_path = None, 
    text_x = None # Position to plot actual label
):
    nrows, ncols = subplot_size 
    num_of_labels = len(labels)
    assert len(img_paths) == num_of_labels, 'img_paths and labels must have same number of items'
    assert nrows * ncols <= num_of_labels, f'nrows * ncols must be <= {num_of_labels}'
    fontdict = {
        'fontproperties': FontProperties(fname=font_path),
        'fontsize': 18,
        'color': 'black',
        'verticalalignment': 'top',
        'horizontalalignment': 'left'
    }

    plt.figure(figsize=figsize)
    for i in range(min(nrows * ncols, num_of_labels)):
        plt.subplot(nrows, ncols, i + 1)
        image, label = plt.imread(img_paths[i]), labels[i]
        plt.imshow(image)

        fontdict['color'] = 'black'  # Reset the color
        if pred_labels: draw_predicted_text(label, pred_labels[i], fontdict, text_x)
        else: plt.text(text_x, 0, '\n'.join(label), fontdict=fontdict)
        plt.axis('off')

    if legend_loc and annotate_loc and pred_labels:
        plt.subplots_adjust(left=0, right=0.75)
        plt.legend(handles=[
            Patch(color='green', label='Full match'),
            Patch(color='dodgerblue', label='Character match'),
            Patch(color='red', label='Wrong prediction'),
            Patch(color='black', label='Actual character'),
            Patch(color='gray', label='Missing position'),
        ], loc=legend_loc)

        annotate_text = [f'{idx + 1:02d}. {text}' for idx, text in enumerate(pred_labels)]
        plt.annotate(
            f'Model predictions:\n{chr(10).join(annotate_text)}',
            fontproperties = FontProperties(fname=font_path),
            xycoords = 'axes fraction',
            fontsize = 14,
            xy = annotate_loc,
        )


def plot_training_results(history, save_name, figsize=(16, 14), subplot_size=(2, 2)):
    nrows, ncols = subplot_size
    if 'lr' in history.keys(): del history['lr']
    assert nrows * ncols <= len(history), f'nrows * ncols must be <= {len(history)}'
    fig = plt.figure(figsize=figsize)

    for idx, name in enumerate(history):
        if 'val' in name: continue
        plt.subplot(nrows, ncols, idx + 1)
        plt.plot(history[name], linestyle='solid', marker='o', color='crimson', label='Train')
        plt.plot(history[f'val_{name}'], linestyle='solid', marker='o', color='dodgerblue', label='Validation')
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel(name, fontsize=14)

        title = name.replace('acc', 'accuracy')\
                    .replace('seq_', 'sequence_')\
                    .replace('char_', 'character_')\
                    .replace('lev_', 'levenshtein_')\
                    .replace('edit_', 'levenshtein_')\
                    .replace('_', ' ').capitalize()
        plt.title(title, fontsize=18)
        plt.legend(loc='best')

    fig.savefig(save_name, bbox_inches='tight')
    plt.show()
