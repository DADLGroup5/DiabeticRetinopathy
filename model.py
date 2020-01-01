model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=X_train[0].shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.20))

model.add(Conv2D(32, (3, 3), padding='same'))
#model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())
#model.add(Dropout(0.10))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(512, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024))
model.add(Dropout(0.25))
model.add(Dense(activation='softmax', units=2))

model.compile(loss='sparse_categorical_crossentropy', optimizer = opt, metrics=["accuracy"])
model.summary()