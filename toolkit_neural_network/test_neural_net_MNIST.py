from neural_net_MNIST import create_ann, train, test


ann, test_data = train(create_ann())
test(ann, test_data)
