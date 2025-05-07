import numpy as np

pong_list = [
    -19.01834862385321,
    -18.8411214953271,
    -19.110091743119266,
    -19.38053097345133,
    -18.897196261682243,
    -19.0,
    -19.0188679245283,
    -18.935185185185187,
    -18.97222222222222,
    -18.833333333333332
]

adapted_pong_list = [
    333.92070484581495,
    342.5389755011136,
    328.66379310344826,
    332.57080610021785,
    335.9170305676856,
    353.1890660592255,
    342.28187919463085,
    330.5739514348786,
    345.55555555555554,
    339.56043956043953
]

def print_result(norm_list, adapted_list):
    print(np.mean(norm_list), np.std(norm_list))
    print(np.mean(adapted_list), np.std(adapted_list))

def main():
    print_result(pong_list, adapted_pong_list)

if __name__ == "__main__":
    main()