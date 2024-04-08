import ssl
import tensorflow as tf
import tensorflow_datasets as tfds
from keras_unet.models import custom_unet



ssl._create_default_https_context = ssl._create_unverified_context

dataset, info = tfds.load('oxford_iiit_pet', split='train', with_info=True)
print(info)


def preprocess_data(sample):
    image = tf.image.resize(sample['image'], (256, 256))
    mask = tf.image.resize(sample['segmentation_mask'], (256, 256))
    image = tf.cast(image, tf.float32) / 255.0
    mask = tf.cast(mask, tf.float32) / 255.0
    return image, mask


dataset = dataset.map(preprocess_data)
train_dataset = dataset.take(3000).shuffle(1000).batch(32)
test_dataset = dataset.skip(3000).batch(32)

model = custom_unet(
    input_shape=(256, 256, 3),
    use_batch_norm=True,
    num_classes=1,
    filters=64,
    dropout=0.2,
    output_activation='sigmoid'
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=3, validation_data=test_dataset)
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')






