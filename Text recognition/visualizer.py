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
        text_y = end * label_length * 3
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
                plt.text(text_x + 20, text_y, actual_char, fontdict=fontdict)
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
    figsize = (15, 7),
    subplot_size = (2, 8), # tuple: (rows, columns) to display
    show_legend = True, # Only for predictions
    font_path = None, 
    text_x = None # Position of actual label to plot
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

    if show_legend and pred_labels is not None:
        plt.subplots_adjust(left=0, right=0.75)
        plt.legend(handles=[
            Patch(color='green', label='Full match'),
            Patch(color='dodgerblue', label='Character match'),
            Patch(color='red', label='Wrong prediction'),
            Patch(color='black', label='Actual character'),
            Patch(color='gray', label='Missing position'),
        ], loc=(2.7, 1.75))

        annotate_text = [f'{idx + 1:02d}. {text}' for idx, text in enumerate(pred_labels)]
        plt.annotate(
            f'Model predictions:\n{chr(10).join(annotate_text)}',
            fontproperties = FontProperties(fname=font_path),
            xycoords = 'axes fraction',
            fontsize = 14,
            xy = (2.8, 0),
        )


def plot_training_results(history, save_name, edist_metric_name='edist', figsize=(15, 5)):
    fig = plt.figure(figsize=figsize)
    edist, val_edist = history[edist_metric_name], history[f'val_{edist_metric_name}']
    loss, val_loss = history['loss'], history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(edist, linestyle='solid', marker='o', color='crimson', label='Train')
    plt.plot(val_edist, linestyle='solid', marker='o', color='dodgerblue', label='Validation')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Mean Edit Distance', fontsize=14)
    plt.title('Average Levenshtein Distance', fontsize=18)
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(loss, linestyle='solid', marker='o', color='crimson', label='Train')
    plt.plot(val_loss, linestyle='solid', marker='o', color='dodgerblue', label='Validation')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training Loss', fontsize=18)
    plt.legend(loc='best')

    fig.savefig(save_name, bbox_inches='tight')
    plt.show()
