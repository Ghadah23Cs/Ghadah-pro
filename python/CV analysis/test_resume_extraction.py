import json
from resume_extractor import extract_text_from_pdf, extract_resume_info  
import pymupdf

# 🔹 تحديد مسار ملف السيرة الذاتية
cv_file_path = "C:/Users/TECHNO GATE/Desktop/CV/python/CV analysis/CV.pdf"  

# 🔹 قراءة ملف PDF واستخراج النص
with open(cv_file_path, "rb") as pdf_file:
    text = extract_text_from_pdf(pdf_file)  

# 🔹 استخراج البيانات من النص المستخرج
resume_data = extract_resume_info(text)

# 🔹 طباعة البيانات المستخرجة
print("\n✅ **المعلومات المستخرجة من السيرة الذاتية:**")
print(json.dumps(resume_data, indent=4, ensure_ascii=False))
