import os
from argparse import Namespace

import numpy as np
import httpimport
import torch
import torch.optim as optim
from tqdm import tqdm as tqdm_notebook
import matplotlib.pyplot as plt

# import data preprocessing and modeling functions from https://github.com/jasoriya/CS6120-PS2-support/tree/master/utils
with httpimport.remote_repo(['data_vectorization', 'model', 'helper'],
                            'https://raw.githubusercontent.com/jasoriya/CS6120-PS2-support/master/utils/'):
    from data_vectorization import Vocabulary, SequenceVocabulary, SurnameVectorizer, SurnameDataset, generate_batches
    from model import SurnameGenerationModel, sample_from_model, decode_samples
    from helper import make_train_state, update_train_state, normalize_sizes, compute_accuracy, sequence_loss, \
        set_seed_everywhere, handle_dirs


def fit_nn(rnn):
    args = Namespace(
        # Data and Path information
        surname_csv="https://raw.githubusercontent.com/jasoriya/CS6120-PS2-support/master/data/surnames/surnames_with_splits.csv",
        vectorizer_file="vectorizer.json",
        model_state_file="model.pth",
        save_dir="./" + str(rnn),  # give path here
        # Model hyper parameters
        char_embedding_size=32,
        rnn_hidden_size=rnn,  # give hidden size
        # Training hyper parameters
        seed=1337,
        learning_rate=0.001,
        batch_size=128,
        num_epochs=100,
        early_stopping_criteria=5,
        # Runtime options
        catch_keyboard_interrupt=True,
        cuda=True,
        expand_filepaths_to_save_dir=True,
        reload_from_files=False,
    )

    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir,
                                            args.vectorizer_file)

        args.model_state_file = os.path.join(args.save_dir,
                                             args.model_state_file)

        print("Expanded filepaths: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.model_state_file))

    # Check CUDA
    if not torch.cuda.is_available():
        args.cuda = False

    args.device = torch.device("cuda" if args.cuda else "cpu")

    print("Using CUDA: {}".format(args.cuda))

    # Set seed for reproducibility
    set_seed_everywhere(args.seed, args.cuda)

    # handle dirs
    handle_dirs(args.save_dir)
    if args.reload_from_files:
        # training from a checkpoint
        dataset = SurnameDataset.load_dataset_and_load_vectorizer(args.surname_csv,
                                                                  args.vectorizer_file)
    else:
        # create dataset and vectorizer
        dataset = SurnameDataset.load_dataset_and_make_vectorizer(args.surname_csv)
        dataset.save_vectorizer(args.vectorizer_file)

    vectorizer = dataset.get_vectorizer()

    model = SurnameGenerationModel(char_embedding_size=args.char_embedding_size,
                                   char_vocab_size=len(vectorizer.char_vocab),
                                   rnn_hidden_size=args.rnn_hidden_size,
                                   padding_idx=vectorizer.char_vocab.mask_index)
    mask_index = vectorizer.char_vocab.mask_index

    model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min', factor=0.5,
                                                     patience=1)
    train_state = make_train_state(args)

    epoch_bar = tqdm_notebook(desc='training routine',
                              total=args.num_epochs,
                              position=0)

    dataset.set_split('train')
    train_bar = tqdm_notebook(desc='split=train',
                              total=dataset.get_num_batches(args.batch_size),
                              position=1,
                              leave=True)
    dataset.set_split('val')
    val_bar = tqdm_notebook(desc='split=val',
                            total=dataset.get_num_batches(args.batch_size),
                            position=1,
                            leave=True)

    try:
        for epoch_index in range(args.num_epochs):
            train_state['epoch_index'] = epoch_index

            # Iterate over training dataset

            # setup: batch generator, set loss and acc to 0, set train mode on
            dataset.set_split('train')
            batch_generator = generate_batches(dataset,
                                               batch_size=args.batch_size,
                                               device=args.device)
            running_loss = 0.0
            running_acc = 0.0
            model.train()

            for batch_index, batch_dict in enumerate(batch_generator):
                # the training routine is these 5 steps:

                # --------------------------------------
                # step 1. zero the gradients
                optimizer.zero_grad()

                # step 2. compute the output
                y_pred = model(x_in=batch_dict['x_data'])
                # print("x:", batch_dict['x_data'][0][0])
                # print("x:", batch_dict['x_data'][0])
                # print("x:", batch_dict['x_data'])

                # print("x:", y_pred[0][0])
                # print("x:", y_pred[0])
                # print("x:", y_pred)

                # print("y:", y_pred)

                # step 3. compute the loss
                loss = sequence_loss(y_pred, batch_dict['y_target'], mask_index)

                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                optimizer.step()
                # -----------------------------------------
                # compute the  running loss and running accuracy
                running_loss += (loss.item() - running_loss) / (batch_index + 1)
                acc_t = compute_accuracy(y_pred, batch_dict['y_target'], mask_index)
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                # update bar
                train_bar.set_postfix(loss=running_loss,
                                      acc=running_acc,
                                      epoch=epoch_index)
                train_bar.update()

            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)

            # Iterate over val dataset

            # setup: batch generator, set loss and acc to 0; set eval mode on
            dataset.set_split('val')
            batch_generator = generate_batches(dataset,
                                               batch_size=args.batch_size,
                                               device=args.device)
            running_loss = 0.
            running_acc = 0.
            model.eval()

            for batch_index, batch_dict in enumerate(batch_generator):
                # compute the output
                y_pred = model(x_in=batch_dict['x_data'])

                # step 3. compute the loss
                loss = sequence_loss(y_pred, batch_dict['y_target'], mask_index)

                # compute the  running loss and running accuracy
                running_loss += (loss.item() - running_loss) / (batch_index + 1)
                acc_t = compute_accuracy(y_pred, batch_dict['y_target'], mask_index)
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                # Update bar
                val_bar.set_postfix(loss=running_loss, acc=running_acc,
                                    epoch=epoch_index)
                val_bar.update()

            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)

            train_state = update_train_state(args=args, model=model,
                                             train_state=train_state)

            scheduler.step(train_state['val_loss'][-1])

            if train_state['stop_early']:
                break

            # move model to cpu for sampling
            model = model.cpu()
            sampled_surnames = decode_samples(
                sample_from_model(model, vectorizer, num_samples=2),
                vectorizer)
            epoch_bar.set_postfix(sample1=sampled_surnames[0],
                                  sample2=sampled_surnames[1])
            # move model back to whichever device it should be on
            model = model.to(args.device)

            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.update()

    except KeyboardInterrupt:
        print("Exiting loop")
    np.random.choice(np.arange(len(vectorizer.nationality_vocab)), replace=True, size=2)
    model.load_state_dict(torch.load(train_state['model_filename']))
    print(train_state['model_filename'])
    model = model.to(args.device)

    dataset.set_split('test')
    batch_generator = generate_batches(dataset,
                                       batch_size=args.batch_size,
                                       device=args.device)
    running_acc = 0.
    # running_loss = 0.
    model.eval()
    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        # # for i in range(0,19):
        # print(batch_index)
        # print(batch_dict)

        y_pred = model(x_in=batch_dict['x_data'])
        # print(y_pred[0][0][0])
        # print(y_pred[0][0])
        # print(len(y_pred[0]))
        # print(y_pred)

        # perplexity+=math.log(y_pred)
        # compute the loss
        loss = sequence_loss(y_pred, batch_dict['y_target'], mask_index)

        # compute the accuracy
        running_loss += (loss.item() - running_loss) / (batch_index + 1)

        acc_t = compute_accuracy(y_pred, batch_dict['y_target'], mask_index)
        running_acc += (acc_t - running_acc) / (batch_index + 1)

    final_perplex = torch.exp(torch.tensor(running_loss))
    train_state['test_loss'] = running_loss
    train_state['test_acc'] = running_acc
    train_tensor = train_state['train_loss']
    validation_tensor = train_state['val_loss']
    print("Test loss: {};".format(train_state['test_loss']))
    print("Train perplexity;", train_tensor[-1])
    print("Validation perplexity;", validation_tensor[-1])
    print("Test perplexity: {};".format(final_perplex.item()))  # compute and print perplexity here
    print("Test Accuracy: {}".format(train_state['test_acc']))
    entire_corpus = (final_perplex.item() + train_tensor[-1] + validation_tensor[-1]) / 3
    print("Perplexity of the entire corpus:", entire_corpus)
    model.load_state_dict(torch.load(train_state['model_filename']))
    model = model.to(args.device)

    ##############  OVER ENTIRE TEST SET ##############################################
    # dataset.set_split('test')
    # batch_generator = generate_batches(dataset,
    #                               batch_size=args.batch_size,
    #                               device=args.device)
    # running_acc = 0.
    # # running_loss = 0.
    # perplexity_character_dict ={}
    # accuracy_character_dict={}
    # # compute the output
    # model.eval()
    # enumerated = list(enumerate(batch_generator))

    # for i in range(1,20):
    #     for batch_index, batch_dict in enumerated:
    #     # for j in range(0,len(batch_dict['x_data'])):
    #         y_pred = model(x_in=batch_dict['x_data'][:,0:i])

    #           # compute the loss
    #         loss = sequence_loss(y_pred, batch_dict['y_target'][:,0:i], mask_index)

    #         # compute the accuracy
    #         running_loss += (loss.item() - running_loss) / (batch_index + 1)
    #         acc_t = compute_accuracy(y_pred, batch_dict['y_target'][:,0:i], mask_index)
    #         running_acc += (acc_t - running_acc) / (batch_index + 1)
    #     accuracy_character_dict[i] = running_acc
    #     perplexity_character_dict[i] = torch.exp(torch.tensor(running_loss)).item()
    # print(accuracy_character_dict)
    # print(perplexity_character_dict)

    ##############  OVER ENTIRE TEST SET ##############################################

    ##############  FOR ONE SURNAME AND UNCOMMENT THIS ##############################################

    obj = SurnameVectorizer(vectorizer.char_vocab, vectorizer.nationality_vocab)
    from_vect, to_vect = obj.vectorize('Singhrathore', 19)
    from_vect = from_vect.reshape(19, 1)
    to_vect = to_vect.reshape(19, 1)
    from_tensor = torch.from_numpy(from_vect).to(args.device)
    to_tensor = torch.from_numpy(to_vect).to(args.device)
    running_acc = 0.
    # running_loss = 0.
    perplexity_character_dict = {}
    accuracy_character_dict = {}
    # compute the output
    # enumerated = list(enumerate(batch_generator))
    for i in range(1, 20):
        # for batch_index, batch_dict in enumerated:
        y_pred = model(from_tensor[0:i])

        # compute the loss
        loss = sequence_loss(y_pred, to_tensor[0:i], mask_index)

        # compute the accuracy
        running_loss += (loss.item() - running_loss)

        acc_t = compute_accuracy(y_pred, to_tensor[0:i], mask_index)
        running_acc += (acc_t - running_acc)
        accuracy_character_dict[i] = running_acc
        perplexity_character_dict[i] = torch.exp(torch.tensor(running_loss)).item()
    print(accuracy_character_dict)
    print(perplexity_character_dict)

    ##############  FOR ONE SURNAME AND UNCOMMENT THIS ##############################################
    # number of names to generate
    num_names = 10
    model = model.cpu()
    # Generate nationality hidden state
    sampled_surnames = decode_samples(
        sample_from_model(model, vectorizer, num_samples=num_names),
        vectorizer)
    # Show results
    print("-" * 15)
    for i in range(num_names):
        print(sampled_surnames[i])
    return perplexity_character_dict, accuracy_character_dict


def begin(listRnn):
    ultimate_dict = {}
    for entry in listRnn:
        perplex_dict, acc_dict = fit_nn(entry)
        ultimate_dict[entry] = perplex_dict
        y1 = list(perplex_dict.values())
        x1 = list(perplex_dict.keys())
        plt.plot(x1, y1, label="RNN="+str(entry))
    plt.xlabel('Number of characters observed')
    plt.ylabel('Peplexity')
    plt.title('Perplexity VS Character')
    plt.legend()
    plt.show()




temp = [32,64]
begin(temp)
