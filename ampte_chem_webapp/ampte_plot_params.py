from pydantic import ConfigDict, BaseModel, field_validator
from streamlit.runtime.uploaded_file_manager import UploadedFile


class PlotParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    xlo: float
    xhi: float
    ylo: float
    yhi: float
    bin_width: float
    tic: int = 0
    zmin: float = 0.
    zmax: float = 0
    mmpq_log: bool = True
    normalize: bool = False
    br0max: float | str = ''
    br1max: float | str = ''
    br2max: float | str = ''
    # trend_item: str = 'L'
    trend_items: list[str] = ['L']
    # csv_file: UploadedFile | None = None
    csv_files: list[UploadedFile] | None = None

    @field_validator('xlo', 'xhi', 'ylo', 'yhi', 'bin_width', 'tic', 'zmin', 'zmax', 'br0max', 'br1max', 'br2max',
                     mode='before')
    def strip_str(cls, value) -> str:
        if isinstance(value, str):
            return value.strip()
        return value

    # class Config:
    #     arbitrary_types_allowed = True

    # @validator('tic', allow_reuse=True)
    # def valid_tic(cls, value):
    #     if value <= 0:
    #         return 50
    #     return value

# class ETPlotParams(BaseModel):
#     xlo: float = 0.
#     xhi: float = 500.
#     ylo: float = 0.
#     yhi: float = 500.
#     bin_width: float = 50.
#     zmax: float = 0
#     tic: int = 50
#
#     @validator('bin_width')
#     @classmethod
#     def valid_et_binw(cls, value):
#         if value < 0.5:
#             return 1
#         return value
#
#     @validator('tic')
#     @classmethod
#     def valid_tic(cls, value):
#         if value <= 0:
#             return 50
#         return value
#
#
# class MMqPlotParams(BaseModel):
#     xlo: float = 0.5
#     xhi: float = 100.
#     ylo: float = 0.5
#     yhi: float = 100.
#     bin_width = .01
#     zmax: float = 0.
#
#     @validator('xlo', 'ylo')
#     def valid_xylo(cls, value):
#         if value <= 0:
#             return 0.1
#         return value
#
#     @validator('bin_width')
#     def valid_mmq_binw(cls, value):
#         if value <= 0:
#             return .01
#         return value
#
#
# class MEPlotParams(BaseModel):
#     xlo: int = 0
#     xhi: int = 96
#     ylo: int = 0
#     yhi: int = 28
#     zmin: float = 1.
#     zmax: float = 0.
#
#
# class TrendPlotParams(BaseModel):
#     trend_item: str
#
