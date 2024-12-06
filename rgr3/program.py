import mnist_loader
from network import Network

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = Network([784, 30, 10])

net.SGD(
    training_data=training_data, 
    epochs=30, 
    mini_batch_size=10, 
    eta=3.0, 
    test_data=test_data
    )

validation_data = list(validation_data)

validation_inputs = [x for x, _ in validation_data]
validation_labels = [y for _, y in validation_data]

predicted_classes = net.predict(validation_inputs)

correct_predictions = sum(int(pred == actual) for pred, actual in zip(predicted_classes, validation_labels))
accuracy = correct_predictions / len(validation_labels)

print(f"Точность на валидационном наборе: {accuracy:.2%}")