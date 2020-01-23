import wandb


class Logger:
    def __init__(self, cnfg):
        super().__init__()
        wandb.init(
            name=cnfg['logger']['run'],
            project=cnfg['logger']['project'],
            config=cnfg
        )

    def log_train(self, epoch, loss, accuracy, label):
        print("\n[INFO][TRAIN] \t Train results: \t \
           Loss:  {}, \t Acc: {}".format(loss, accuracy))
        wandb.log({'Train Loss': loss}, commit=False, step=epoch)
        wandb.log({'Train Accuracy': accuracy}, commit=False, step=epoch)

    def log_test(self, step, loss, accuracy, label):
        print("[INFO][TEST] \t Test results: \t \
           Loss:  {}, \t Acc: {} \n".format(loss, accuracy))
        wandb.log({'Test Loss': loss}, commit=False, step=step)
        wandb.log({'Test Accuracy': accuracy}, commit=False, step=step)

    def log_model(self, pth):
        wandb.save(pth)
