import pytorch_lightning as pl
from utils.joystick_data_loader import JoystickDataModule
from utils.joystick_pressure_model import JoystickPressureModel
from multiprocessing import freeze_support
from pytorch_lightning.loggers import CSVLogger


if __name__ == '__main__':
    freeze_support()

    seq_len = 100
    pred_distance = 1000
    n_port = 4

    csv_logger = CSVLogger("logs", name="test_est_pressure", version=n_port)

    model = JoystickPressureModel.load_from_checkpoint(checkpoint_path="logs/create_est_pressure/version_1/checkpoints/epoch=0-step=6306.ckpt", hidden_size=1024)
    trainer = pl.Trainer(accelerator='cuda', logger=csv_logger)

    test_data = JoystickDataModule(dataset_path='result_peak_data.pkl', seq_len=seq_len, pred_distance=pred_distance,
                                   batch_size=1000, n_of_worker=8, test_data_port_n=n_port)

    trainer.test(model, datamodule=test_data)

