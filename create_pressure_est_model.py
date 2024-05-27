import pytorch_lightning as pl
from utils.joystick_data_loader import JoystickDataModule
from utils.joystick_pressure_model import JoystickPressureModel
from multiprocessing import freeze_support
from pytorch_lightning.loggers import CSVLogger

if __name__ == '__main__':
    freeze_support()

    csv_logger = CSVLogger("logs", name="create_est_pressure", version=1)

    seq_len = 100
    pred_distance = 1000

    data = JoystickDataModule(dataset_path='result_peak_data.pkl', seq_len=seq_len, pred_distance=pred_distance,
                              batch_size=500, n_of_worker=8)
    model = JoystickPressureModel(hidden_size=1024, num_layers=1, learning_rate=0.001)

    trainer = pl.Trainer(accelerator='gpu', devices='auto', max_epochs=1, enable_progress_bar=True, logger=csv_logger)
    trainer.fit(model=model, datamodule=data)

