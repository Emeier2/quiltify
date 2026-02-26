from pydantic import BaseModel


class GuideSection(BaseModel):
    title: str
    content: str


class QuiltingGuide(BaseModel):
    title: str
    sections: list[GuideSection]
    raw_text: str = ""
