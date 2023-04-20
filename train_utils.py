from click import progressbar
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix




def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # if not title:
    #     if normalize:
    #         title = 'Normalized confusion matrix'
    #     else:
    #         title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        None
        # print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots(figsize=(27, 27))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           # xticklabels=classes, yticklabels=classes,
           xticklabels=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36','37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60'],
           yticklabels=classes,
           title=title,
           # ylabel='True label',
           # xlabel='Predicted label'
           )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0.01:
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-.')
    # plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(bottom=0.15)
    return ax


def calculate_accuracy(Y_hat, Y):
    _, Y_hat = Y_hat.max(1)
    trues = (Y_hat == Y.long()) * 1
    trues = trues.cpu().numpy()

    accuracy = np.mean(trues)

    return accuracy, Y_hat.cpu().numpy(), Y.cpu().numpy()


def evaluate_set(model, model_type, data_loader, output_folder, set_name):
    model.eval()

    average_accuracy = 0
    n_samples = 0

    y_true = []
    y_pred = []

    for batch_idx, batch in enumerate(data_loader):
        print(str(batch_idx) + " / " + str(len(data_loader)))
        X = batch[0]
        Y = batch[1].to(device)

        batch_size = Y.shape[0]
        n_samples += batch_size

        if model_type == "CNN3D":
            X = prime_X_cnn3d(X).to(device)

        elif model_type == "FUSION":
            X = prime_X_fusion(X, model.use_pose, model.use_ir)

        out = model(X)

        accuracy, Y_hat, Y = calculate_accuracy(out, Y)
        average_accuracy += accuracy * batch_size

        y_true.append(Y)
        y_pred.append(Y_hat)

        # print([classes[i] for i in np.int32(Y_hat)])
        # print([classes[i] for i in np.int32(Y)])
        # print()

        batch_log = open(output_folder + "batch_log.txt", "a+")
        batch_log.write("[" + str(set_name) + " - " + str(batch_idx) + "/" + str(len(data_loader)) +
                        "] Accuracy : " + str(accuracy))
        batch_log.write("\r\n")
        batch_log.close()

    return average_accuracy / n_samples, y_true, y_pred


def train_model_new(model,
                    model_type,
                    optimizer,
                    learning_rate,
                    weight_decay,
                    gradient_threshold,
                    epochs,
                    accumulation_steps,
                    evaluate_test,
                    output_folder,
                    train_generator,
                    test_generator,
                    validation_generator = None):

    # Lists for plotting
    time_batch = []
    time_epoch = [0]
    loss_batch = []
    loss_epoch = []

    train_errors = []

    # Accumulation of values if updating gradients over multiple batches
    accuracy_accumulated = 0
    loss_accumulated = 0

    if optimizer == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        print("Optimizer not recognized ... exit()")
        exit()

    for e in range(epochs):
        model.train()
        errors_temp = []

        start = time.time()

        start_batch = time.time()
        for batch_idx, batch in enumerate(train_generator):
            # BATCH TRAINING
            print(str(e) + " - " + str(float(batch_idx / accumulation_steps)) +
                  "/" + str(int(len(train_generator) / accumulation_steps)))
            X = batch[0]
            Y = batch[1].to(device)

            if model_type == "CNN3D":
                X = prime_X_cnn3d(X)

            elif model_type == "FUSION":
                X = prime_X_fusion(X, model.use_pose, model.use_ir)

            out = model(X)

            loss = F.cross_entropy(out, Y.long()) / accumulation_steps
            loss_accumulated += loss.item()
            loss.backward()

            # Gradient clipping
            if gradient_threshold > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_threshold)

            # Accuracy over batch
            accuracy_batch, _, _ = calculate_accuracy(out, Y)
            accuracy_accumulated += accuracy_batch / accumulation_steps

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()

                # Save loss per batch
                time_batch.append(e + (batch_idx / accumulation_steps) / (len(train_generator) / accumulation_steps))
                loss_batch.append(loss.item())

                batch_log = open(output_folder + "batch_log.txt", "a+")
                batch_log.write("[" + str(e) + " - " + str(int(batch_idx / accumulation_steps)) + "/"
                                + str(int(len(train_generator) / accumulation_steps)) +
                                "] Accuracy : " + str(accuracy_accumulated) + ", loss : " + str(loss_accumulated))
                batch_log.write("\r\n")
                batch_log.close()
                errors_temp.append(1 - accuracy_accumulated)

                print("Batch took : " + str(time.time() - start_batch) + "s")

                accuracy_accumulated = 0
                loss_accumulated = 0
                start_batch = time.time()

        # VALIDATION STEP
        if validation_generator is not None:
            with torch.no_grad():
                validation_accuracy, _, _ = evaluate_set(model,
                                                         model_type,
                                                         validation_generator,
                                                         output_folder,
                                                         "VAL")

        # TEST STEP
        if evaluate_test:
            with torch.no_grad():
                test_accuracy, _, _ = evaluate_set(model,
                                                   model_type,
                                                   test_generator,
                                                   output_folder,
                                                   "TEST")

        # Save loss per epoch
        time_epoch.append(e + 1)
        loss_epoch.append(
            sum(loss_batch[e * len(train_generator): (e + 1) * len(train_generator)]) / len(train_generator))

        # Average accuracy over epoch
        train_errors.append(np.mean(errors_temp))

        # Write log data
        # Log file (open and close after each epoch so we can read realtime
        end = time.time()
        log = open(output_folder + "log.txt", "a+")
        log.write("Epoch : " + str(e) + ", err train : " + str(np.mean(errors_temp)))
        if validation_generator is not None:
            log.write(", val accuracy : " + str(validation_accuracy))
        if evaluate_test:
            log.write(", test accuracy : " + str(test_accuracy) + " ")
        log.write("in : " + str(end - start) + " seconds")
        log.write("\r\n")
        log.close()

        # Save model
        torch.save(model.state_dict(), str(output_folder) + "model" + str(e) + ".pt")

    return model
