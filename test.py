from math import log


class Judger:
    # Initialize Judger, with the path of accusation list and law articles list
    def __init__(self):
        self.result = {"cnt": 0, "score": 0}

    # Gen new results according to the truth and users output
    def gen_new_result(self, truth_tensor, label_tensor):
        """
        :param truth: tensor (bsz)  fact
        :param label: tensor (bsz)  users output
        :return: count one result score done
        """
        bsz = truth_tensor.shape[0]
        for i in range(bsz):
            truth = int(truth_tensor[i])
            label = int(label_tensor[i])
            self.result["cnt"] += 1
            sc = 0
            if truth == 301:
                if label == 301:
                    sc = 1
            elif truth == 302:
                if label == 302:
                    sc = 1
            else:
                if label > 300:
                    sc = 0
                else:
                    v1 = truth
                    v2 = label
                    v = abs(log(v1 + 1) - log(v2 + 1))
                    if v <= 0.2:
                        sc = 1
                    elif v <= 0.4:
                        sc = 0.8
                    elif v <= 0.6:
                        sc = 0.6
                    elif v <= 0.8:
                        sc = 0.4
                    elif v <= 1.0:
                        sc = 0.2
                    else:
                        sc = 0
            sc = sc * 1.0
            self.result["score"] += sc

    # Generatue all scores
    def get_score(self):
        s3 = 1.0 * self.result["score"] / self.result["cnt"]
        return s3
