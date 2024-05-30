to obtain result and save, run the command to train:
python main_rl.py > sac_res.txt 2>&1
python main_rl.py > dqn_res.txt 2>&1
python main_rl.py > dqn_res_1_step.txt 2>&1
and then run the command to do simulations:
python rl_long_simulation.py --output_root "C:\Users\Qiyuan Huang\Desktop\cxsj\pto-selfish-mining\experiments\sac\logs" --load_experiment "BitcoinFeeModel(0.35, 0.5, 10, 10, 0.01, 10)_20240330-165127"
or to store the result in a .txt file:
python rl_long_simulation.py --output_root "C:\Users\Qiyuan Huang\Desktop\cxsj\pto-selfish-mining\experiments\sac" --load_experiment "logs" > "C:\Users\Qiyuan Huang\Desktop\cxsj\pto-selfish-mining\experiments\sac\result.txt" 2>&1
to run the log to graph:
python rl_log_to_graph.py "C:\Users\Qiyuan Huang\Desktop\cxsj\pto-selfish-mining\experiments\sac\logs\BitcoinFeeModel(0.35, 0.5, 10, 10, 0.01, 10)_20240330-165127\log.txt" > "C:\Users\Qiyuan Huang\Desktop\cxsj\pto-selfish-mining\experiments\sac\graph.txt" 2>&1
to train the original model, run:
python rl_long_simulation.py --output_root "C:\Users\Qiyuan Huang\Desktop\cxsj\pto-selfish-mining\experiments" --load_experiment "sac" > "C:\Users\Qiyuan Huang\Desktop\cxsj\pto-selfish-mining\experiments\origin\result.txt" 2>&1
