import numpy as np

def compute_mean_std(data, str):
    # 计算均值
    returns_mean = np.mean(data)
    print(str + " Mean: ", returns_mean)

    # 计算标准差
    returns_std = np.std(data)
    print(str + " Standard Deviation: ", returns_std)

def main():

    # qbert_returns = [5962.251655629139, 6298.013245033113, 5944.833333333333, 6042.114093959732, 6358.04794520548, 
    #            5837.903225806452, 6002.649006622517, 5857.792207792208, 6748.591549295775, 5928.9473684210525]
    # adapted_qbert_returns = [7199.094202898551, 6837.234042553191, 6627.027027027027, 6678.671328671328, 
    #                    6957.857142857143, 6320.689655172414, 6607.413793103448, 7368.613138686132,
    #                    6415.0, 6704.340277777777]
    # breakout_returns = [32.5, 31.825, 37.689655172413794, 37.52272727272727, 32.46774193548387, 33.492063492063494, 
    #                     39.43589743589744, 32.59722222222222, 33.97727272727273, 32.92982456140351]
    # adapted_breakout_returns = [36.86206896551724, 36.55882352941177, 28.8, 33.36666666666667, 34.03448275862069,
    #                             34.724137931034484, 30.23076923076923, 32.37735849056604, 33.703125, 30.8]
    
    # pong_returns = [15.947368421052632, 14.964912280701755, 15.517857142857142, 14.351851851851851, 14.436363636363636, 14.818181818181818, 
    #                     15.789473684210526, 14.872727272727273, 14.607142857142858, 14.092592592592593]
    # adapted_pong_returns = [15.87719298245614, 15.321428571428571, 15.464285714285714, 16.283333333333335, 16.338983050847457,
                                # 15.810344827586206, 16.916666666666668, 14.41818181818181, 16.694915254237287, 15.75438596491228]
    
    # # Qbert/1%/CQL_ln_tta/eval_tent
    # qbert_eval_tent_returns = [6176.986754966887, 6678.846153846154, 5874.5161290322585, 5941.118421052632, 5710.294117647059,
    #                  6409.523809523809, 5927.5, 5716.612903225807, 5782.258064516129, 6004.391891891892]
    # qbert_eval_tent_adapted_returns = [233.7874659400545, 216.80555555555554, 369.36619718309856, 555.6497175141243, 445.57377049180326, 
    #                               495.73770491803276, 345.26143790849676, 270.2724358974359, 594.9004975124378, 287.1951219512195]

    # # Qbert/1%/CQL_ln_tta/eval_entropy0.1
    qbert_eval_entropy01_returns = [6946.6783216783215, 6044.701986754967, 7762.592592592592, 6600.844594594595, 7188.204225352113,
                                  7057.857142857143, 6771.551724137931, 6490.2027027027025, 6833.333333333333, 6670.973154362416]
    # # Qbert/1%/CQL_ln_tta/eval_entropy0.2
    qbert_eval_entropy02_returns = [7018.9655172413795, 6062.664473684211, 7162.410071942446, 6174.666666666667, 7053.417266187051,
                                    6989.43661971831, 6470.529801324504, 6873.287671232877, 7023.415492957746, 6861.979166666667]
    # # Qbert/1%/CQL_ln_tta/eval_entropy0.3
    qbert_eval_entropy03_returns = [7180.31914893617, 5833.870967741936, 6373.310810810811, 6540.10067114094, 6103.476821192053,
                                    6011.019736842105, 7284.53237410072, 6904.895104895105, 5431.944444444444, 5789.649681528663]
     # # Qbert/1%/CQL_ln_tta/eval_entropy0.1_epsilon0.01
    qbert_eval_entropy03_epsilon001_returns = [6125.324675324675, 6600.838926174497, 6571.527777777777, 6452.210884353742, 5984.455128205128,
                                    6225.167785234899, 5930.921052631579, 7001.630434782609, 5990.384615384615, 6191.216216216216]
    
    # compute_mean_std(breakout_returns, "breakout normal")
    # compute_mean_std(adapted_breakout_returns, "breakout adapted")

    # compute_mean_std(qbert_returns, "qbert normal")
    # compute_mean_std(adapted_qbert_returns, "qbert adapted")

    # compute_mean_std(pong_returns, "pong normal")
    # compute_mean_std(adapted_pong_returns, "pong adapted")

    # compute_mean_std(qbert_eval_tent_returns, "qbert normal")
    # compute_mean_std(qbert_eval_tent_adapted_returns, "qbert adapted")

    # compute_mean_std(qbert_eval_entropy01_returns, "qbert adapted")
    # compute_mean_std(qbert_eval_entropy02_returns, "qbert adapted")
    # compute_mean_std(qbert_eval_entropy03_returns, "qbert adapted")
    compute_mean_std(qbert_eval_entropy03_epsilon001_returns, "qbert adapted")

if __name__ == "__main__":
    main()