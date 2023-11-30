# run a command on each thread using processpool

import os
from concurrent.futures import ProcessPoolExecutor

def commandGenerator(dataset, iid, scaffold, weighted, mix):
    command = f"python federated_main.py --model=cnn " + \
        f"--dataset={dataset} --iid={iid} --epochs=100 --local_ep=5 --local_bs=16 " + \
        f"--frac=1 --num_users=10 --scaffold={scaffold} --weighted={weighted} --mix={mix}"
    
    return command

if(__name__ == "__main__"):
    commands = []
    options = []
    datasets = ["cifar", "mnist"]
    algorithms = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for dataset in datasets:
        commands.append(commandGenerator(dataset, 1, 0, 0, 0)) # baseline
        options.append([dataset, 1, 0, 0, 0])
        # noniid
        for algorithm in algorithms:
            commands.append(commandGenerator(dataset, 0, algorithm[0], algorithm[1], 0))
            options.append([dataset, 0, algorithm[0], algorithm[1], 0])
        # transformed
        for algorithm in algorithms:
            commands.append(commandGenerator(dataset + "_transformed", 0, algorithm[0], algorithm[1], 2))
            options.append([dataset + "_transformed", 0, algorithm[0], algorithm[1], 2])
        # noniid-mixed
        for algorithm in algorithms:
            commands.append(commandGenerator(dataset, 0, algorithm[0], algorithm[1], 1))
            options.append([dataset, 0, algorithm[0], algorithm[1], 1])
        # transformed-noniid-mixed
        for algorithm in algorithms:
            commands.append(commandGenerator(dataset + "_transformed", 0, algorithm[0], algorithm[1], 1))
            options.append([dataset + "_transformed", 0, algorithm[0], algorithm[1], 1])

    # transformed baseline
    commands.append(commandGenerator("mnist_transformed", 1, 0, 0, 0))
    options.append(["mnist_transformed", 1, 0, 0, 0])
    commands.append(commandGenerator("cifar_transformed", 1, 0, 0, 0))
    options.append(["cifar_transformed", 1, 0, 0, 0])

    if (len(commands) != 36):
        print("Error: number of commands is not 36")
        exit(1)

    with ProcessPoolExecutor(max_workers=36) as executor:
        index = 0
        for command in commands:
            # write output to a file, filename is options
            filename = "_".join([str(option) for option in options[index]])
            filename = "../save/outputs/" + filename + ".txt"
            command = command + " > " + filename
            executor.submit(os.system, command)
            index += 1
    


