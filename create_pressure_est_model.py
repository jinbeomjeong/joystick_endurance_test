import pytorch_lightning as pl
from utils.joystick_data_loader import JoystickDataModule
from utils.joystick_pressure_model import JoystickPressureModel
from multiprocessing import freeze_support
from pytorch_lightning.loggers import CSVLogger

if __name__ == '__main__':
    freeze_support()

    csv_logger = CSVLogger("logs", name="est_pressure_model", version=1)

    seq_len = 1000

    data = JoystickDataModule(dataset_path='result_peak_data.pkl', seq_len=seq_len, forecast_distance=1000,
                              batch_size=1000, n_of_worker=16)
    model = JoystickPressureModel(hidden_size=128, learning_rate=0.003)

    trainer = pl.Trainer(accelerator='gpu', devices='auto', max_epochs=5, enable_progress_bar=True, logger=csv_logger)
    trainer.fit(model=model, datamodule=data)

