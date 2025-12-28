from pydantic import BaseModel, Field
from typing import Optional


from pydantic import BaseModel, Field


class ApartmentFeatures(BaseModel):
    material: str = Field(..., description="Тип дома")
    metro: str = Field(..., description="Станция метро")
    address: str = Field(..., description="Адрес квартиры")
    rooms: float = Field(..., gt=0, description="Количество комнат")
    area_total: float = Field(..., gt=0, description="Общая площадь, м²")
    area_living: float = Field(..., ge=0, description="Жилая площадь, м²")
    area_kitchen: float = Field(..., ge=0, description="Площадь кухни, м²")
    floor: int = Field(..., ge=0, description="Этаж")
    total_floors: int = Field(..., gt=0, description="Этажность дома")
    year_built: int = Field(..., gt=1900, description="Год постройки")
    time_to_metro: float = Field(..., ge=0, description="Время до метро, минут")
    is_first_floor: int = Field(..., ge=0, le=1, description="1 — первый этаж")
    is_last_floor: int = Field(..., ge=0, le=1, description="1 — последний этаж")

    living_share: float | None = None
    kitchen_share: float | None = None
    floor_ratio: float | None = None

    def model_post_init(self, __context):
        self.living_share = self.area_living / self.area_total
        self.kitchen_share = self.area_kitchen / self.area_total
        self.floor_ratio = self.floor / self.total_floors

class PriceResponse(BaseModel):
    predicted_price: float | int = Field(..., description="Предсказанная цена")
    currency: str = "RUB"