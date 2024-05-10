import pytorch_lightning as pl
from utils.joystick_data_loader import JoystickDataModule
from utils.joystick_pressure_model import JoystickPressureModel
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    data = JoystickDataModule(dataset_path='result_peak_data.pkl', batch_size=256, n_of_worker=2)
    model = JoystickPressureModel()

    trainer = pl.Trainer(accelerator='gpu', devices='auto', max_epochs=3, enable_progress_bar=True)
    trainer.fit(model=model, datamodule=data)

