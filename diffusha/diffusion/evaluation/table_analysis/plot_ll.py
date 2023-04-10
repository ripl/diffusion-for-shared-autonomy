import matplotlib.pyplot as plt
import numpy as np
from csv_2_dict import csv_2_dict, Entry


def set_fig():
    # fig setting
    import yaml
    from yaml.loader import SafeLoader
    # Open the file and load the file
    with open('matplotlibrc.yaml') as f:
        plt_std = yaml.load(f, Loader=SafeLoader)
    plt.style.use('seaborn-whitegrid')
    palette = plt.get_cmap('Set1')
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 13,
             }

    plt.rcParams["figure.figsize"] = [6.00, 6.00]
    plt.rcParams["figure.autolayout"] = True
    for key, value in plt_std.items():
        plt.rcParams[key] = value


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="Landing",
        choices=["Landing", "Reaching"],
        help="Task name to visualize",
    )
    parser.add_argument(
        "--config", default="0215", help="Specific configs for this task"
    )
    args = parser.parse_args()

    task = args.task
    config = args.config

    # for p_laggy and p_noisy
    sample_list = [0.1, 0.2, 0.3, 0.4]
    used_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    single_list = [0.8]

    dicts = []
    curr_list = used_list
    for i in range(3):
        curr_dic = csv_2_dict(f"summary/{task}-{config}-{i}.csv")
        dicts.append(curr_dic)
    forward = [i / 10.0 for i in range(0, 11)]

    for p in curr_list:
        Laggy = p
        Noisy = p
        assisted = True
        actors = ['noisy', 'laggy']
        success_plot = True
        bench_mean = True
        for index, actor in enumerate(actors):
            plt.cla()

            ep = [[] for _ in range(len(forward))]
            diff = [[] for _ in range(len(forward))]
            avg = [[] for _ in range(len(forward))]

            success = [[] for _ in range(len(forward))]
            crash = [[] for _ in range(len(forward))]
            floating = [[] for _ in range(len(forward))]
            bench_success = []
            diff_model_success = []
            bench_crash = []
            diff_model_crash = []
            bench_float = []
            diff_model_float = []

            if actor == 'noisy':
                p = Noisy
            elif actor == 'laggy':
                p = Laggy
            else:
                pass

            if actor == 'benchmark':
                assisted = False
                actor = 'expert'
                plot_label = 'benchmark(expert with no assistance)'

            elif actor == '~di_na':
                assisted = False
                plot_label = 'copilot(diffusion model)'
            else:
                assisted = True
                plot_label = actor

            title = f'{task.lower()}_rates_{actor}_{int(p * 10)}'

            for index, f in enumerate(forward):
                generate_key = str(f) + str(p) + actor + str(assisted)
                bench_entry = dicts[0][str(f) + str(p) + 'expert' + 'False']
                diff_model_entry = dicts[0][str(f) + str(p) + '~di_na' + 'False']

                bench_success.append(bench_entry.success[0])
                diff_model_success.append(diff_model_entry.success[0])
                bench_crash.append(bench_entry.game_over[0])
                diff_model_crash.append(diff_model_entry.game_over[0])
                bench_float.append(bench_entry.floating[0])
                diff_model_float.append(diff_model_entry.floating[0])

                for dic in dicts:
                    entry: Entry = dic[generate_key]

                    ep[index].append(entry.ep_length[0])
                    diff[index].append(entry.action_diffs[0])
                    avg[index].append(entry.reward[0])

                    success[index].append(entry.success[0])
                    crash[index].append(entry.game_over[0])
                    floating[index].append(entry.floating[0])

            if bench_mean:
                bench_success = np.array(bench_success).mean()
                bench_success = [bench_success for _ in range(len(forward))]
                diff_model_success = np.array(diff_model_success).mean()
                diff_model_success = [diff_model_success for _ in range(len(forward))]

                bench_crash = np.array(bench_crash).mean()
                bench_crash = [bench_crash for _ in range(len(forward))]
                diff_model_crash = np.array(diff_model_crash).mean()
                diff_model_crash = [diff_model_crash for _ in range(len(forward))]

                bench_float = np.array(bench_float).mean()
                bench_float = [bench_float for _ in range(len(forward))]
                diff_model_float = np.array(diff_model_float).mean()
                diff_model_float = [diff_model_float for _ in range(len(forward))]

            if success_plot:
                labels = ['Success', 'Float', 'Crash']
                colors = ['#1F449C', '#A8B6CC', '#F05039']
                for index, attr in enumerate([success, floating, crash]):
                    mean = np.array(attr).mean(axis=1)
                    var = np.array(attr).std(axis=1)

                    # print selected p_laggy / p_noisy statistic result
                    """
                    print(f"{actor}:", labels[index], ": ",
                          "$", "%.2f" % mean[0], "\pm", "%.2f" % var[0], "$ & ",
                          "$", "%.2f" % mean[4], "\pm", "%.2f" % var[4], "$ & ",
                          "$", "%.2f" % mean[-1], "\pm", "%.2f" % var[-1], "$")
                    """
                    r1 = list(map(lambda x: x[0] - x[1], zip(mean, var)))
                    r2 = list(map(lambda x: x[0] + x[1], zip(mean, var)))
                    plt.plot(forward, mean, color=colors[index], label=labels[index], marker='o', linewidth=3.0)
                    plt.fill_between(forward, r1, r2, color=colors[index], alpha=0.17)

                plt.plot(forward, np.array(bench_success), color=colors[0], label='_nolegend_', linewidth=2.0,
                         linestyle='--', alpha=1.0)
                plt.plot(forward, np.array(diff_model_success), color=colors[0], label='_nolegend_', linewidth=2.0,
                         linestyle=':', alpha=1.0)
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.xlabel(r"Forward Diffusion Ratio: $\gamma$")
            plt.xlim((0, 1.0))
            plt.ylim((0, 100))

            # plt.ylabel(attr)
            #                 plt.savefig(f"0215/{task}/{title}.pdf",format='pdf')
            plt.show()
