import re
import pymupdf  # استيراد pymupdf بدلاً من fitz
import spacy
import csv
import nltk

nltk.download('punkt')

# تحميل نماذج spaCy
nlp = spacy.load('en_core_web_sm')
nlp_skills = spacy.load('services/skills_ner_model')

# ---------------------------------- تحميل الكلمات المفتاحية ----------------------------------
def load_keywords(file_path):
    """ تحميل الكلمات المفتاحية من ملف CSV """
    with open(file_path, 'r', encoding="utf-8") as file:
        reader = csv.reader(file)
        return set(row[0] for row in reader)

# ---------------------------------- استخراج النص من PDF ----------------------------------
def extract_text_from_pdf(uploaded_file):
    """ استخراج النص من ملف PDF باستخدام pymupdf """
    try:
        doc = pymupdf.open(stream=uploaded_file.read(), filetype="pdf")  # استخدام pymupdf
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"❌ خطأ في استخراج النص من PDF: {e}")
        return ""

# ---------------------------------- استخراج الاسم ----------------------------------
def extract_name(doc):
    """ استخراج الاسم من المستند باستخدام NER """
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            names = ent.text.split()
            if len(names) >= 2 and names[0].istitle() and names[1].istitle():
                return names[0], ' '.join(names[1:])
    return "", ""

# ---------------------------------- استخراج البريد الإلكتروني ----------------------------------
def extract_email(text):
    """ استخراج البريد الإلكتروني باستخدام regex """
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return email_match.group(0) if email_match else "Not Found"

# ---------------------------------- استخراج رقم الهاتف ----------------------------------
def extract_phone(text):
    """ استخراج رقم الهاتف باستخدام regex """
    phone_match = re.search(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", text)
    return phone_match.group(0) if phone_match else "Not Found"

# ---------------------------------- استخراج الدرجة العلمية ----------------------------------
def extract_major(text):
    """ استخراج الدرجة العلمية من النص بناءً على كلمات مفتاحية """
    major_keywords = load_keywords('python/CV analysis/majors.csv')
    for keyword in major_keywords:
        if keyword.lower() in text.lower():
            return keyword
    return "Not Found"

# ---------------------------------- استخراج المهارات ----------------------------------

# ---------------------------------- تحميل كلمات المهارات من CSV ----------------------------------
def load_keywords(file_path):
    """ تحميل قائمة المهارات من ملف CSV وتحويلها إلى مجموعة Set لسهولة البحث """
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            reader = csv.reader(file)
            return {row[0].strip().lower() for row in reader if row}
    except Exception as e:
        print(f"❌ خطأ في تحميل كلمات المهارات من CSV: {e}")
        return set()

# تحميل المهارات من CSV مرة واحدة فقط
skills_keywords = load_keywords('python/CV analysis/newSkills.csv')

# ---------------------------------- استخراج المهارات من CSV ----------------------------------
def csv_skills(text):
    """ البحث عن المهارات في النص بناءً على الكلمات المفتاحية """
    text_lower = text.lower()
    return {skill for skill in skills_keywords if skill in text_lower}

# ---------------------------------- استخراج المهارات باستخدام NER ----------------------------------
def extract_skills_from_ner(text):
    """ استخراج المهارات باستخدام نموذج NER المدرب """
    if nlp_skills is None:
        return set()

    doc = nlp_skills(text)
    skills = {ent.text.strip() for ent in doc.ents if ent.label_ == 'SKILL'}
    return skills

# ---------------------------------- تصفية المهارات ----------------------------------
def is_valid_skill(skill_text):
    """ التأكد من أن المهارة صالحة وليست مجرد رقم أو كلمة غير واضحة """
    return len(skill_text) > 1 and any(char.isalpha() for char in skill_text)

# ---------------------------------- استخراج جميع المهارات ----------------------------------
def extract_skills(text):
    """ استخراج المهارات من CSV و NER ثم تصفيتها """
    try:
        skills_csv = csv_skills(text)
        skills_ner = extract_skills_from_ner(text)

        # تصفية المهارات غير الصالحة
        filtered_skills_csv = {skill for skill in skills_csv if is_valid_skill(skill)}
        filtered_skills_ner = {skill for skill in skills_ner if is_valid_skill(skill)}

        # دمج النتائج بدون تكرار
        combined_skills = filtered_skills_csv.union(filtered_skills_ner)
        return list(combined_skills)
    except Exception as e:
        print(f"❌ خطأ أثناء استخراج المهارات: {e}")
        return []

# ---------------------------------- استخراج الخبرة ----------------------------------
def extract_experience(text):
    """ تحليل النص واستخراج مستوى الخبرة """
    verbs = [token.text for token in nlp(text) if token.pos_ == 'VERB']
    
    senior_keywords = {'lead', 'manage', 'direct', 'supervise'}
    mid_keywords = {'develop', 'analyze', 'coordinate'}
    junior_keywords = {'assist', 'support', 'collaborate'}

    if any(keyword in verbs for keyword in senior_keywords):
        return "Senior"
    elif any(keyword in verbs for keyword in mid_keywords):
        return "Mid-Level"
    elif any(keyword in verbs for keyword in junior_keywords):
        return "Junior"
    return "Entry Level"

# ---------------------------------- الدالة الرئيسية لاستخراج المعلومات ----------------------------------
def extract_resume_info(text):
    """ استخراج جميع المعلومات من النص """
    doc = nlp(text)

    return {
        "first_name": extract_name(doc)[0],
        "last_name": extract_name(doc)[1],
        "email": extract_email(text),
        "phone": extract_phone(text),
        "degree_major": extract_major(text),
        "skills": extract_skills(text),
        "experience": extract_experience(text)
    }
