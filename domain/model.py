from dataclasses import dataclass
import datetime
from enum import Enum


class MachineType(Enum):
    MotorAc = "Motor AC"
    MotorDc = "Motor DC"
    Turbine = "Turbine"
    
class MachineSize(Enum):
    Small = "Small"
    Medium = "Medium"
    Large = "Large"

class MeasurementType(Enum):
    Acceleration = "Acceleration"
    Velocity = "Velocity"
    Displacement = "Displacement"

class FaultFrequencyType(Enum):
    Bearing = "Bearing"
    Gear = "Gear"
    Shaft = "Shaft"

@dataclass
class FaultFrequencyModel:
    name: str
    type: FaultFrequencyType
    fault_frequency_in_orders: float
 
@dataclass
class MachineModel:
    name: str
    type: MachineType
    size: MachineSize
    fault_frequencies: list[FaultFrequencyModel]

@dataclass
class SpectrumPeak:
    order: float
    amplitude: float

@dataclass
class Spectrum:
    timestamp: datetime
    fmax_order: float
    peaks: list[SpectrumPeak]
    total_energy: float
    last_overall_rms: list[float]
    waveform_crest_factor: float

@dataclass
class Measurement:
    type: MeasurementType
    value: Spectrum
    machine: MachineModel

