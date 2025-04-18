from enum import Enum

class MongoCollection(Enum):
    # ChatHistory = "chat_history"
    Course = "courses_2024_2025"
    Program = "programs_2024_2025"
    General = "general_2024_2025"

class MongoIndex(Enum):
    FULL_TEXT = "full_text_index"
    VECTOR = "vector_index"

class CourseLevel(Enum):
    LEVEL_000 = "000"
    LEVEL_100 = "100"
    LEVEL_200 = "200"
    LEVEL_300 = "300"
    LEVEL_400 = "400"
    LEVEL_500 = "500"
    LEVEL_600 = "600"
    LEVEL_700 = "700"
    LEVEL_800 = "800"
    LEVEL_A00 = "a00"
    LEVEL_C00 = "c00"
    LEVEL_D00 = "d00"
    LEVEL_E00 = "e00"
    LEVEL_G00 = "g00"
    LEVEL_H00 = "h00"
    LEVEL_J00 = "j00"
    LEVEL_K00 = "k00"
    LEVEL_L00 = "l00"
    LEVEL_M00 = "m00"
    LEVEL_N00 = "n00"
    LEVEL_P00 = "p00"
    LEVEL_S00 = "s00"
    LEVEL_T00 = "t00"
    LEVEL_W00 = "w00"

class AcademicLevel(Enum):
    ALL = "all"
    UGRAD = "undergraduate"
    GRAD = "graduate"

class Degree(Enum):
    MASTER_OF_EDUCATION = "Master of Education"
    GRAD_CERT_TEACHING_ESL = "Graduate Certificate Teaching ESL"
    BSCNUR = "BSCNUR"
    C_OSI = "C-OSI"
    MASTER_SCI_APPL_PHYS_THER = "Master of Sci Appl, Phys Ther"
    PC_BUV = "PC-BUV"
    PC_EHC = "PC-EHC"
    BSC_ARCHITECTURE = "Bachelor of Science (Architecture)"
    GC_HSE = "GC-HSE"
    C_INLL = "C-INLL"
    BNUR = "BNUR"
    BCLJD = "BCLJD"
    GC_DHS = "GC-DHS"
    C_CIT = "C-CIT"
    D_HRDS = "D-HRDS"
    GC_DDM = "GC-DDM"
    GC_ACN = "GC-ACN"
    CERT_PROF_ENGL_PROF_COMM = "Certificate Prof Engl for Prof Comm"
    MIST = "MIST"
    PC_DCM = "PC-DCM"
    GD_OMD = "GD-OMD"
    BSCRS = "BSCRS"
    GRAD_CERT_AIR_SPACE_LAW = "Graduate Certificate in Air and Space Law"
    GC_ISN = "GC-ISN"
    GC_SPR = "GC-SPR"
    BA_ED = "BA-ED"
    GRAD_CERT_MARKETING = "Graduate Certificate in Marketing"
    C_TMH = "C-TMH"
    D_MHNP = "D-MHNP"
    P_PHD_GRAD_DIP_CHILD_PSY = "P PhD Graduate Diploma Sc/App Child Psy"
    CERT_FIRST_NATIONS_INUIT_SPS = "Certificate - First Nations and Inuit Student Personnel Services"
    C_STEM = "C-STEM"
    DIP_PUBLIC_RELATIONS = "Diploma in Public Relations and Communications Management"
    C_PACC = "C-PACC"
    C_IKM = "C-IKM"
    GRAD_CERT_LIBRARY_INFO = "Graduate Certificate in Library & Info St"
    C_DAM = "C-DAM"
    GC_CYS = "GC-CYS"
    DIP_METEOROLOGY = "Diploma in Meteorology"
    CERT_HEALTH_SOCIAL_SERVICES = "Certificate in Health & Social Services Management"
    GRAD_CERT_BIOTECHNOLOGY = "Graduate Certificate in Biotechnology"
    MDCM = "MDCM"
    CERT_PROF_FRENCH = "Certificate - Proficiency in French"
    D_PDNP = "D-PDNP"
    CERT_FOOD_SCIENCE = "Certificate in Food Science"
    GC_AVL = "GC-AVL"
    MASTER_OF_MUSIC = "Master of Music"
    GD_ACN = "GD-ACN"
    MASTER_OF_MANAGEMENT = "Master of Management"
    GRAD_CERT_HR_MGM = "Graduate Certificate in Hum Resources Mgm"
    BSC_NUTRITIONAL_SCIENCES = "Bachelor of Science (Nutritional Sciences)"
    GC_NUR = "GC-NUR"
    GC_PAA = "GC-PAA"
    MPP = "MPP"
    CERT_HR_MANAGEMENT = "Certificate in Human Resources Management"
    CERT_EDU_FIRST_NATIONS = "Certificate in Education - First Nations and Inuit"
    BACHELOR_OF_SCIENCE = "Bachelor of Science"
    GC_GPR = "GC-GPR"
    GC_EIM = "GC-EIM"
    BSC_FOOD_SCIENCE = "Bachelor of Science (Food Science)"
    GRAD_CERT_BIOINFORMATICS = "Graduate Certificate Bioinformatics"
    D_ONCO = "D-ONCO"
    CERT_PR_COMM_MANAGEMENT = "Certificate in Public Relations and Communications Management"
    C_RMED = "C-RMED"
    BACHELOR_OF_EDUCATION = "Bachelor of Education"
    CERT_ABORIGINAL_LITERACY = "Certificate - Aboriginal Literacy Education"
    GRAD_CERT_NURSING = "Graduate Certificate in Nursing"
    BED_CERTIFIED_TEACHERS = "Bachelor of Education for Certified Teachers"
    PC_EHA = "PC-EHA"
    FARM_MANAGEMENT_TECH = "Farm Management Technology"
    MASTER_OF_BUSINESS_ADMIN = "Master of Business Admin"
    CERT_INCLUSIVE_EDUCATION = "Certificate - Inclusive Education"
    C_OMS = "C-OMS"
    DOCTOR_OF_CIVIL_LAW = "Doctor of Civil Law"
    BSC_KINESIOLOGY = "Bachelor of Science (Kinesiology)"
    BACHELOR_OF_ENGINEERING = "Bachelor of Engineering"
    C_PAG = "C-PAG"
    C_SUIN = "C-SUIN"
    BACHELOR_OF_SOCIAL_WORK = "Bachelor of Social Work"
    GRAD_DIP_NURSING = "Graduate Diploma in Nursing"
    GRAD_CERT_COUNS_TEACHING = "Graduate Certificate Couns Appl to Teaching"
    PC_DAB = "PC-DAB"
    MASTER_OF_SACRED_THEOLOGY = "Master of Sacred Theology"
    GC_HMG = "GC-HMG"
    C_FSSS = "C-FSSS"
    CERT_MANAGEMENT = "Certificate in Management"
    CERT_ECOLOGICAL_AGRICULTURE = "Certificate in Ecological Agriculture"
    CERT_SUPPLY_CHAIN_MGT = "Certificate in Supl Chain Mgt & Logis"
    GC_HRA = "GC-HRA"
    DOCTOR_OF_PHILOSOPHY = "Doctor of Philosophy"
    D_PERF = "D-PERF"
    C_ACYB = "C-ACYB"
    C_EDL3 = "C-EDL3"
    C_TBME = "C-TBME"
    GRAD_CERT_EDU_LEADERSHIP1 = "Graduate Certificate Educational Leadersh 1"
    BEDVOC = "BEDVOC"
    CERT_APPLIED_MARKETING = "Certificate in Applied Marketing"
    PC_PMG = "PC-PMG"
    BACHELOR_OF_MUSIC = "Bachelor of Music"
    CERT_FIRST_NATIONS_EDU_LEADERSHIP = "Certificate - First Nations and Inuit Educational Leadership"
    DIP_MANAGEMENT = "Diploma in Management"
    GRAD_CERT_EDU_LEADERSHIP2 = "Graduate Certificate Educational Leadersh 2"
    GC_ACC = "GC-ACC"
    GRAD_CERT_POST_MBA = "Graduate Certificate Post MBA"
    DMD = "DMD"
    BSC_AGRICULTURAL_ENV = "Bachelor of Science (Agricultural and Environmental Sciences)"
    C_IAD = "C-IAD"
    DIP_ENVIRONMENT = "Diploma in Environment"
    BACHELOR_OF_ARTS_SCIENCE = "Bachelor of Arts and Science"
    GRAD_DIP_MINING_ENG = "Graduate Diploma in Mining Engineering"
    GD_LGT = "GD-LGT"
    GRAD_DIP_REG_DIET = "Graduate Diploma in Reg Diet (R.D.) Cred"
    C_PERF = "C-PERF"
    GC_NIT = "GC-NIT"
    GC_FTY = "GC-FTY"
    MASTER_OF_SCIENCE_APPLIED = "Master of Science Applied"
    GC_BMG = "GC-BMG"
    BACHELOR_OF_COMMERCE = "Bachelor of Commerce"
    C_TPED = "C-TPED"
    PC_AMC = "PC-AMC"
    PC_PLG = "PC-PLG"
    GC_FAN = "GC-FAN"
    GC_PNP = "GC-PNP"
    D_SUIN = "D-SUIN"
    GC_DAC = "GC-DAC"
    C_EDIL = "C-EDIL"
    GC_ABM = "GC-ABM"
    DOCTOR_OF_MUSIC = "Doctor of Music"
    CERT_PUBLIC_RELATIONS = "Certificate in Public Relations"
    GRAD_CERT_ENTREPRENEURSHIP = "Graduate Certificate in Entrepreneurship"
    PGD_AT = "PGD-AT"
    C_PSWC = "C-PSWC"
    C_ACFI = "C-ACFI"
    MASTER_OF_SOCIAL_WORK = "Master of Social Work"
    D_MRP = "D-MRP"
    GRAD_DIP_CLINICAL_RESEARCH = "Graduate Diploma in Clinical Research"
    CERT_PROF_FRENCH_PROF_COMM = "Certificate Prof French for Prof Comm"
    GRAD_CERT_CHRONIC_PAIN = "Graduate Certificate in Chronic Pain Mgmt"
    MASTER_OF_ARCHITECTURE = "Master of Architecture"
    MASTER_OF_URBAN_PLANNING = "Master of Urban Planning"
    BACHELOR_OF_THEOLOGY = "Bachelor of Theology"
    MASTER_SCI_APPL_OCC_THER = "Master of Sci Appl, Occ Ther"
    BENG_BIORESOURCE = "Bachelor of Engineering (Bioresource)"
    MASTER_OF_ARTS = "Master of Arts"
    MASTER_OF_ENGINEERING = "Master of Engineering"
    GC_BTR = "GC-BTR"
    BGE = "BGE"
    GC_LTR = "GC-LTR"
    CERT_PROF_ENGLISH = "Certificate - Proficiency in English"
    C_IBM = "C-IBM"
    GC_MRA = "GC-MRA"
    GC_PAG = "GC-PAG"
    GC_PRC = "GC-PRC"
    C_INMS = "C-INMS"
    PC_BUA = "PC-BUA"
    GRAD_CERT_COMPARATIVE_LAW = "Graduate Certificate in Comparative Law"
    C_DRR = "C-DRR"
    C_PBPC = "C-PBPC"
    GC_DSN = "GC-DSN"
    GRAD_CERT_PR_MGM = "Graduate Certificate in Pub Relations Mgm"
    MASTER_OF_LAWS = "Master of Laws"
    LICENTIATE_IN_MUSIC = "Licentiate in Music"
    MASTER_OF_SCIENCE = "Master of Science"
    PC_PLM = "PC-PLM"
    D_CCAN = "D-CCAN"
    BACHELOR_OF_ARTS = "Bachelor of Arts"

class Faculty(Enum):
    EMPTY = "isEmpty"
    UNKNOWN = "00"
    AGRICULTURAL_ENVIRONMENTAL_SCIENCES = "Agricultural & Environmental Sciences"
    DESAUTELS_MANAGEMENT = "Desautels Faculty of Management"
    ARTS = "Faculty of Arts"
    DENTAL_MEDICINE = "Faculty of Dental Medicine and Oral Health Sciences"
    EDUCATION = "Faculty of Education"
    ENGINEERING = "Faculty of Engineering"
    LAW = "Faculty of Law"
    MEDICINE_HEALTH_SCIENCES = "Faculty of Medicine and Health Sciences"
    SCIENCE = "Faculty of Science"
    GRADUATE_STUDIES = "Graduate Studies"
    INTERFACULTY_STUDIES = "Interfaculty Studies"
    INTERFACULTY_BA_SC = "Interfaculty, B.A. & Sc."
    POST_GRADUATE_DENTAL = "Post Graduate Dental Medicine and Oral Health Sciences"
    CONTINUING_STUDIES = "School of Continuing Studies"
    CONTINUING_STUDIES_NON_TR = "School of Continuing Studies (Non-Tr)"
    ENVIRONMENT = "School of Environment"
    NURSING = "School of Nursing"
    PHYSICAL_OCCUPATIONAL_THERAPY = "School of Physical & Occupational Therapy"
    SCHULICH_MUSIC = "Schulich School of Music"
    YR = "YR"

class Department(Enum):
    UNDECLARED = "Undeclared"
    GEOGRAPHY = "Geography"
    FRENCH_LANGUAGE_LITERATURE = "French Language & Literature"
    HISTORY_CLASSICAL_STUDIES = "History and Classical Studies"
    KINESIOLOGY_PHYSICAL_ED = "Kinesiology and Physical Ed"
    PARASITOLOGY = "Parasitology"
    SOCIAL_WORK = "Social Work"
    PEDIATRICS = "Pediatrics"
    HUMAN_GENETICS = "Human Genetics"
    CHEMICAL_ENGINEERING = "Chemical Engineering"
    ECONOMICS = "Economics"
    OBSTETRICS_GYNECOLOGY = "Obstetrics & Gynecology"
    EPIDEMIOLOGY_BIOSTATISTICS = "Epidemiology and Biostatistics"
    NEUROSCIENCE_INTEGRATED_PGM = "Neuroscience, Integrated Pgm"
    AIR_SPACE_LAW = "Air and Space Law"
    ANTHROPOLOGY = "Anthropology"
    ATMOSPHERIC_OCEANIC_SCIENCES = "Atmospheric & Oceanic Sciences"
    EDUCATIONAL_COUNSELLING_PSYCH = "Educational&Counselling Psych"
    FOOD_SCIENCE_AGR_CHEMISTRY = "Food Science&Agr.Chemistry"
    LANGUAGE_INTERCULTURAL_COMM = "Language & Intercultural Comm."
    MCGILL_WRITING_CENTRE = "McGill Writing Centre"
    INSTITUTE_AEROSPACE_ENG = "Institute for Aerospace Eng."
    EARTH_PLANETARY_SCIENCES = "Earth & Planetary Sciences"
    GLOBAL_STRATEGIC_COMM = "Global & Strategic Comm."
    FRENCH_LANGUAGE_CENTRE = "French Language Centre"
    PATHOLOGY = "Pathology"
    INGRAM_SCHOOL_NURSING = "Ingram School of Nursing"
    DIAGNOSTIC_RADIOLOGY = "Diagnostic Radiology"
    BIELER_SCHOOL_ENVIRONMENT = "Bieler School of Environment"
    ART_HISTORY_COMMUNICATIONS = "Art History & Communications"
    CHEMISTRY = "Chemistry"
    TECHNOLOGY_INNOVATION = "Technology & Innovation"
    MANAGEMENT = "Management"
    GLOBAL_PUBLIC_HEALTH = "Global & Public Health"
    PHYSIOLOGY = "Physiology"
    PERSONAL_CULTURAL_ENRICHMENT = "Personal & Cultural Enrichment"
    GERALD_BRONFMAN_DEPT_ONCOLOGY = "Gerald Bronfman Dept Oncology"
    URBAN_PLANNING = "Urban Planning"
    ELECTRICAL_COMPUTER_ENGR = "Electrical & Computer Engr"
    MICROBIOLOGY_IMMUNOLOGY = "Microbiology & Immunology"
    OCCUPATIONAL_HEALTH = "Occupational Health"
    PHARMACOLOGY_THERAPEUTICS = "Pharmacology and Therapeutics"
    FAMILY_MEDICINE = "Family Medicine"
    CIVIL_ENGINEERING = "Civil Engineering"
    MUSIC_RESEARCH = "Music Research"
    POPULATION_GLOBAL_HEALTH = "Population and Global Health"
    FAC_PARTNER_SUMMER_STUDIES = "Fac Partner & Summer Studies"
    LAW = "Law"
    ARCHITECTURE = "Architecture"
    CAREER_PROFESSIONAL_DEVELOP = "Career & Professional Develop"
    HUMAN_NUTRITION = "Human Nutrition"
    REDPATH_MUSEUM = "Redpath Museum"
    TRANSLATION = "Translation"
    COMPARATIVE_LAW = "Comparative Law"
    ANIMAL_SCIENCE = "Animal Science"
    SCHOOL_PUBLIC_POLICY = "School of Public Policy"
    ANAESTHESIA = "Anaesthesia"
    ANATOMY_CELL_BIOLOGY = "Anatomy and Cell Biology"
    SOCIOLOGY = "Sociology"
    EAST_ASIAN_STUDIES = "East Asian Studies"
    LANGUAGES_LITERATURES_CULTURES = "Languages,Literatures,Cultures"
    PSYCHOLOGY = "Psychology"
    INFORMATION_STUDIES = "Information Studies"
    ENGLISH = "English"
    QUANTITATIVE_LIFE_SCIENCES = "Quantitative Life Sciences"
    ARTS_SCIENCE_ADMIN_SHARED = "Arts & Science Admin (Shared)"
    PHYSICS = "Physics"
    MEDICINE_HEALTH_SCIENCES = "Medicine and Health Sciences"
    RELIGIOUS_STUDIES = "Religious Studies"
    PHYS_OCC_THERAPY = "Phys and Occ Therapy"
    AGRICULTURAL_ECONOMICS = "Agricultural Economics"
    INST_ST_DEVELOPMENT = "Inst for the St of Development"
    FARM_MGMT_TECHNOLOGY_PROGRAM = "Farm Mgmt & Technology Program"
    INTEGRATED_STUDIES_ED = "Integrated Studies in Ed"
    MATHEMATICS_STATISTICS = "Mathematics and Statistics"
    BIOMEDICAL_ENGINEERING = "Biomedical Engineering"
    NATURAL_RESOURCE_SCIENCES = "Natural Resource Sciences"
    BIOCHEMISTRY = "Biochemistry"
    INST_GENDER_SEX_FEM_ST = "Inst for Gender, Sex & Fem St"
    MANAGEMENT_ENTREPRENEURSHIP = "Management & Entrepreneurship"
    ISLAMIC_STUDIES = "Islamic Studies"
    BIOLOGICAL_BIOMEDICAL_ENGR = "Biological & Biomedical Engr"
    PHILOSOPHY = "Philosophy"
    INST_GLOBAL_FOOD_SECURITY = "Inst. of Global Food Security"
    ARTIFICIAL_COURSE = "Artificial Course (not GDEU)"
    ENGINEERING_DEANS_OFFICE = "Engineering - Dean's Office"
    BIORESOURCE_ENGINEERING = "Bioresource Engineering"
    MEDICINE = "Medicine"
    EMPTY = "isEmpty"
    PERFORMANCE = "Performance"
    EDUCATION_DEANS_OFFICE = "Education - Dean's Office"
    DENTAL_MED_ORAL_HEALTH_SCI = "Dental Med & Oral Health Sci"
    ARTS_DEANS_OFFICE = "Arts - Dean's Office"
    AGRICULTURAL_ENV_SC_DEAN = "Agricultural & Env.Sc.-Dean"
    SOCIAL_STUDIES_MEDICINE = "Social Studies of Medicine"
    MINING_MATERIALS_ENGINEERING = "Mining & Materials Engineering"
    SURGERY = "Surgery"
    TROTTIER_INST_SUST_ENG_DESIGN = "Trottier Inst Sust,Eng&Design"
    MECHANICAL_ENGINEERING = "Mechanical Engineering"
    POLITICAL_SCIENCE = "Political Science"
    OTOLARYNGOLOGY_HEAD_NECK_SURG = "Otolaryngology Head/Neck Surg."
    COMMUN_SCIENCES_DISORDERS = "Commun Sciences & Disorders"
    INTERDEPARTMENTAL_STUDIES = "Interdepartmental Studies"
    LINGUISTICS = "Linguistics"
    BIOLOGY = "Biology"
    PSYCHIATRY = "Psychiatry"
    INST_PUBLIC_LIFE_ARTS_IDEAS = "Inst Public Life-Arts & Ideas"
    COMPUTER_SCIENCE = "Computer Science"
    NEUROLOGY_NEUROSURGERY = "Neurology and Neurosurgery"
    MEDICAL_PHYSICS_UNIT = "Medical Physics Unit"
    JEWISH_STUDIES = "Jewish Studies"
    ADMINISTRATION_GOVERNANCE = "Administration & Governance"
    STUDENT_SERVICES = "Student Services"
    SCIENCE = "Science"
    PLANT_SCIENCE = "Plant Science"
    BIOENGINEERING = "Bioengineering"
    HEALTH_SCIENCES_EDUCATION = "Health Sciences Education"
    GRAD_POSTDOC_STUDIES_DEAN = "Grad & Postdoc Studies (Dean)"
    INSTITUTE_STUDY_CANADA = "Institute for Study of Canada"
    OPHTHALMOLOGY = "Ophthalmology"