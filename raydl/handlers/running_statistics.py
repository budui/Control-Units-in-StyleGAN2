import torch


class RunningStatistician:
    _support_statistics = ("min/index", "max/index", "mean", "std")

    def __init__(self, statistic_names=_support_statistics):
        assert all(x in self._support_statistics for x in statistic_names), \
            f"only support statistic in {self._support_statistics}, but got {statistic_names}"
        self.names = statistic_names
        self._dtype = torch.float64

        self._s_counter = 0
        self._s_min, self._s_min_index = None, None
        self._s_max, self._s_max_index = None, None
        self._s_sum = torch.as_tensor(0.0).to(self._dtype)
        self._s_square = torch.as_tensor(0.0).to(self._dtype)

    def sum(self):
        return self._s_sum

    def __len__(self):
        return self._s_counter

    def internal_status(self):
        return {k: v for k, v in self.__dict__.items() if k.startswith("_s_")}

    def reset(self):
        self._s_counter = 0
        self._s_min, self._s_min_index = None, None
        self._s_max, self._s_max_index = None, None
        self._s_sum = torch.as_tensor(0.0).to(self._dtype)
        self._s_square = torch.as_tensor(0.0).to(self._dtype)

    def update(self, x):
        element = torch.as_tensor(x)
        if torch.numel(element) == 0:
            return x

        element = element.detach().flatten().to(self._dtype)
        self._s_counter += int(torch.ones_like(element).sum())

        if "min/index" in self.names:
            min_element = element.min()
            self._s_min, self._s_min_index = (min_element, self._s_counter) if \
                self._s_min is None or self._s_min > min_element else (self._s_min, self._s_min_index)

        if "max/index" in self.names:
            max_element = element.max()
            self._s_max, self._s_max_index = (max_element, self._s_counter) if \
                self._s_max is None or self._s_max < max_element else (self._s_max, self._s_max_index)

        if "mean" in self.names or "std" in self.names:
            self._s_sum += element.sum()

        if "std" in self.names:
            self._s_square += element.square().sum()

    def compute(self):
        amount = int(self._s_counter)
        output = dict()
        for name in self.names:
            if name == "min/index":
                output["min/index"] = (float(self._s_min.cpu()), self._s_min_index)
            if name == "max/index":
                output["max/index"] = (float(self._s_max.cpu()), self._s_max_index)
            if name == "mean":
                if amount == 0:
                    output["mean"] = float("nan")
                else:
                    output["mean"] = float(self._s_sum.cpu()) / amount
            if name == "std":
                if amount == 0:
                    output["std"] = float("nan")
                elif amount == 1:
                    output["std"] = 0.0
                else:
                    mean = output.get("mean", float(self._s_sum.cpu()) / amount)
                    output["std"] = float(((self._s_square - amount * mean ** 2) / (amount - 1)).sqrt().cpu())
        return output
