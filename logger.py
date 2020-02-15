import wandb


class Logger:
    def __init__(self, cnfg):
        super().__init__()
        if cnfg['logger']['wandb']:
            self.dowandb = True
            wandb.init(
                name=cnfg['logger']['run'],
                project=cnfg['logger']['project'],
                config=cnfg,
                reinit=True
            )
        else:
            self.dowandb = False

    def log_train(self, epoch, loss, accuracy, label='train'):
        print("\n[INFO][TRAIN][{}] \t \
           Loss:  {}, \t Acc: {}".format(label, loss, accuracy))
        if self.dowandb:
            wandb.log({'Train Loss': loss}, commit=False, step=epoch)
            wandb.log({'Train Accuracy': accuracy}, commit=False, step=epoch)

    def log_test(self, step, loss, accuracy, label='test'):
        print("[INFO][TEST][{}] \t  \
           Loss:  {}, \t Acc: {} \n".format(label, loss, accuracy))
        if self.dowandb:
            wandb.log({'Test Loss': loss}, commit=False, step=step)
            wandb.log({'Test Accuracy': accuracy}, commit=False, step=step)

    def log_test_adversarial(self, step, loss, accuracy, label='test_adversarial'):
        print("[INFO][TEST][{}] \t \
           Adv Loss:  {}, \t Adv Acc: {} \n".format(label, loss, accuracy))
        if self.dowandb:
            wandb.log({'Test Adversarial Loss': loss}, commit=False, step=step)
            wandb.log({'Test Adversarial Accuracy': accuracy},
                      commit=False, step=step)

    def log_model(self, pth):
        if self.dowandb:
            wandb.save(pth)

    def log_lr(self, values, step):
        if self.dowandb:
            for index, rate in enumerate(values):
                name = "learning_rate_" + str(index)
                wandb.log({name: rate}, commit=False)
