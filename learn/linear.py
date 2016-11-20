from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import train_test_split

from utils.replay import from_local, matrix_window


player, window = 3, 10
nb_classes = 5
nb_epoch = 30
input_dim = (2 * window + 1) ** 2 * 6
if __name__ == '__main__':
    import sys
    replay = from_local(sys.argv[1])
    examples = []
    labels = []
    for frame in range(replay.num_frames - 1):
        replay_frame = replay.get_frame(frame)
        player_y, player_x = replay_frame.player_yx(player)
        player_moves = replay_frame.player_moves(player)
        player_strengths = replay_frame.player_strengths(player)
        player_productions = replay_frame.player_productions(player)
        unowned_strengths = replay_frame.unowned_strengths
        unowned_productions = replay_frame.unowned_productions
        competitor_strengths = replay_frame.competitor_strengths(player)
        competitor_productions = replay_frame.competitor_productions(player)
        for x, y, move in zip(player_x, player_y, player_moves):
            features = [
                matrix_window(player_strengths, x, y, window),
                matrix_window(player_productions, x, y, window),
                matrix_window(unowned_strengths, x, y, window),
                matrix_window(unowned_productions, x, y, window),
                matrix_window(competitor_strengths, x, y, window),
                matrix_window(competitor_productions, x, y, window)
            ]
            examples.append(features)
            labels.append(move)
    X = np.array(examples)
    y = np.array(labels, dtype=int)
    X = X.reshape(len(labels), input_dim)
    X /= 255
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print('#train:', X_train.shape[0])
    print('#test:', X_test.shape[0])

    model = Sequential([
        Dense(nb_classes, input_dim=input_dim),
        Activation('softmax'),
    ])
    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    history = model.fit(X_train, Y_train, nb_epoch=nb_epoch, verbose=1,
                        validation_data=(X_test, Y_test))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
