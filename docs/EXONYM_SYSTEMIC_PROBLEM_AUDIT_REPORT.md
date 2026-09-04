# EXONYM Sistemik Teşhis, Problem Denetimi ve Kusursuzlaştırma Yol Haritası
# (EXONYM Systemic Problem Audit Report & Production-Grade Remediation Blueprint)

**Belge Sürümü / Document Version:** 3.2.0 (Forensic Deep-Research & Fake-Detector Verified Blueprint)  
**Tarih / Date:** 4 Eylül 2026  
**Yetkili Rol / Authoritative Role:** Baş Astrofizik Yazılım Mimarı ve Adli Kod Denetçisi (Principal Astrophysical Software Architect & Scientific Code Auditor)  
**Denetim Motorları & Standartlar:** `/deep-scientific-research`, `/fake-detector`, Statik AST Analizi, Deterministik Regex Taramaları, Doğrulanmış Hakemli Literatür (NASA ADS / DOI), IAU 2015 B3 / CODATA 2018 Fiziksel Sabitleri  
**Hedef Kitle & Yürütücü Ajanlar / Target Execution Agents:** Sistem Operatörü, Otonom Frontier Akıl Yürütme Motorları (GPT-5.6 / SOL)  
**Çalışma Ortamı / Execution Runtime:** Host Windows Git (PowerShell) + WSL Ubuntu Linux 24.04 (Python 3.12.3, `~/.venvs/exonym`)  

---

## 1. Yürütücü Özet ve Mimari Röntgen (Executive Summary & Architectural Roentgen)

EXONYM; NASA TESS (Transiting Exoplanet Survey Satellite) ve gelecekteki uzay tabanlı yüksek hassasiyetli fotometrik misyonlardan bağımsız, tekrarlanabilir ötegezegen keşfi, fiziksel karakterizasyonu ve istatistiki doğrulama (statistical vetting & validation) yürüten, katı izolasyon kurallarına tabi bir astrofizik yazılım çerçevesidir.

Kod tabanının tamamı (`src/exonym/`), anayasal kurallar (`AGENTS.md`), JSON şemaları (`schemas/`), dokümantasyon ağacı (`docs/`, `methods/`), bağımlılık kilitleri (`pyproject.toml`, `requirements-lock.txt`) ve test suitleri (`tests/`) üzerinde gerçekleştirilen `/fake-detector` ve `/deep-scientific-research` denetimleri sonucunda; Mandel & Agol (2002) analitik transit geometrisi, Danby-Halley 3. derece eliptik Kepler denklemi çözücüsü, Wōtan çoklu filtre detrending algoritmaları ve Gaia DR3 astrometrik çapraz eşleşmelerinin matematiksel ve fiziksel olarak doğru çalıştığı teyit edilmiştir.

Buna karşın, EXONYM çerçevesini üretim seviyesinde **"tam otonom, uçtan uca çalışabilir ve adli açıdan kusursuz"** olmaktan alıkoyan **5 kritik mimari darboğaz, eksik otomasyon köprüsü ve adli kod kusuru** tespit edilmiştir:

| Problem Alanı | Adli Kusur Sınıfı | Etkilenen Çekirdek Modüller | Mimari ve Bilimsel Etki Derecesi |
| :--- | :--- | :--- | :--- |
| **1. Eksik Otomasyon Köprüleri** | Eksik CLI & Yükleyici Köprüleri (Hollow Walls) | `vetting/tricera_parse.py`, `localization.py`, `__main__.py` | **KRİTİK (BLOCKED):** `exonym vet` ve `exonym localization` komutlarını yapay `fail-closed` duvarına çarptırır; adayın onaylanmasını kalıcı olarak kilitler. |
| **2. Sayısal Provenans Açıkları** | `/fake-detector` İhlalleri (Choking & Float Truncation) | `vetting/trex/funcs.py`, `vetting/trex/priors.py`, `transit_fit.py` | **YÜKSEK (BIAS):** Düşük kütleli eşlikçiler yapay tabanda yığılarak FPP/NFPP hesaplarını saptırır; analitik integraller kesilmiş ondalıklarla adım süreksizliği üretir. |
| **3. Anayasal Şişkinlik & Format** | Markdown Formatlama & Kural Mükerrerliği | `AGENTS.md:52-58`, `AGENTS.md:15-20`, `AGENTS.md:129` | **ORTA (COGNITIVE):** ~1,500 token'lık gereksiz kural tekrarı ve çıplak terminal komutları ajanlarda bilişsel kargaşa ve linter uyarıları yaratır. |
| **4. Ortam & Lockfile Senkronizasyonu** | Platform ve Python ABI Uyuşmazlığı | `requirements-lock.txt`, `freeze.py:51`, `pyproject.toml` | **YÜKSEK (REPRODUCIBILITY):** WSL Python 3.12 ile Windows Python 3.9 kilit dosyası çelişir; `oktopus` eksikliği stdout/stderr loglarını kirletir. |
| **5. Dokümantasyon Ağacı Kopukluğu** | Yol Uyuşmazlığı & Eksik Operatör Kılavuzu | `docs/README.md`, `docs/EXONYM_SCIENTIFIC_ARCHITECTURE_AND_HOW_TO.md` | **ORTA (USABILITY):** Kök `methods/` bağlantısı tutarsızdır; yeni nesil survey tarama (`harvest`, `auto-vet`, `run-loop`) uçtan uca belgelenmemiştir. |

---

## 2. Derinlemesine Mimari Teşhis (Deep Architectural Analysis)

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             EXONYM SYSTEM ARCHITECTURAL PIPELINE                               │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
   [Target Ingestion & Raw FITS] ───────────────┼──► outputs/archival_vetting_report.json
                                                │    data/raw/tess*tp.fits
                                                │
         ┌──────────────────────────────────────┴──────────────────────────────────────┐
         ▼                                                                            ▼
┌─────────────────────────────────┐                          ┌─────────────────────────────────┐
│       TREX SCENE PIPELINE       │                          │       PRF LOCALIZATION          │
├─────────────────────────────────┤                          ├─────────────────────────────────┤
│ outputs/archival_report.json    │                          │ data/raw/tess*tp.fits (Headers) │
│ data/external/stellar_params    │                          │ Target Coordinates (RA/Dec)     │
│ Gaia DR3 Complete Neighbors     │                          │ MAST PRF Server Grid            │
├─────────────────────────────────┤                          ├─────────────────────────────────┤
│ [MISSING: build-scene CLI]      │                          │ [MISSING: fetch-prf CLI]        │
│   scene_builder.py              │                          │   localization.py fetch/calib   │
├─────────────────────────────────┤                          ├─────────────────────────────────┤
│ data/external/trex_scene.json   │                          │ data/external/tess_prf.fits     │
│   (trex-scene-manifest schema)  │                          │ tess_prf.manifest.json          │
│                                 │                          │ tess_prf.recovery_calibration   │
└────────────────┬────────────────┘                          └────────────────┬────────────────┘
                 │                                                            │
                 ▼                                                            ▼
┌─────────────────────────────────┐                          ┌─────────────────────────────────┐
│   exonym vet (TRICERATOPS)      │                          │  exonym localization (Calib)    │
│   FPP & NFPP Calculation        │                          │  Difference Centroid Residual   │
└─────────────────────────────────┘                          └─────────────────────────────────┘
```

---

### Problem Alanı 1: TREX Sahne ve PRF Şablonu Otomasyon Köprüleri (Missing Scene & PRF Automation Bridges)

#### 1.1 TREX Sahne Yükleyicisi ve Eksik Derleyici Motoru (`build-scene`)
* **Hedef Dosya & Satır Aralıkları:**
  - `src/exonym/vetting/tricera_parse.py:53` (`TREX_SCENE_MANIFEST_RELATIVE_PATH = Path("data") / "external" / "trex_scene.json"`)
  - `src/exonym/vetting/tricera_parse.py:248-390` (`_load_trex_scene(workspace: Any, tic_id: int, sectors: list) -> Tuple[Any, list]`)
  - `src/exonym/vetting/tricera_parse.py:836` (`scene, scene_artifacts = _load_trex_scene(...)`)
  - `src/exonym/vetting/trex/target_scene.py:28-115` (`TargetScene.__init__`)
* **AST ve Hata Yayılım Riski:**
  `_load_trex_scene` fonksiyonu, aday çalışma alanında `data/external/trex_scene.json` dosyasını açmaya çalışır. Dosya mevcut olmadığında veya bozuk olduğunda doğrudan `TrexSceneUnavailableError` fırlatır. Bu hata `tricera_parse.py:840` bloğunda yakalanarak `decisions/triceratops_vetting_decision.json` dosyasına şu değerleri mühürler:
  ```json
  {
    "triage_status": "blocked",
    "execution_status": "unavailable",
    "result_status": "unresolved",
    "claim_eligible": false,
    "fpp": null,
    "nfpp": null
  }
  ```
  Bu durum `exonym vet` komutunun bilimsel hesaplama yapmasını engeller ve pipeline'ı kilitler.
* **Anayasal İhlal:**
  `AGENTS.md` Kural 8 ("Prohibition on Hollow 'Fail-Closed' Walls"). Sistem fonksiyonel bir TRICERATOPS FPP motoruna sahip olmasına rağmen, girdiyi adayın mevcut verilerinden (`outputs/archival_vetting_report.json`, FITS başlıkları ve TIC/Gaia yan ürünleri) derleyen resmi bir CLI motoru bulunmadığı için kullanıcıyı suni bir duvara çarptırmaktadır.
* **Gereken Mimari Çözüm:**
  1. `schemas/trex-scene-manifest.schema.json` dosyasını ve tekerlek fallback eşleniği olan `src/exonym/_resources/schemas/trex-scene-manifest.schema.json` dosyasını Draft 2020-12 standardında oluşturmak.
  2. Adayın arşiv raporundaki Gaia DR3 hedef ve komşu astrometrisini okuyan, yıldız parametrelerini FITS ve TIC kataloglarından çeken, kontrast eğrisi ve arka plan popülasyonunu bağlayan ve SHA-256 özetleriyle mühürleyen `src/exonym/vetting/trex/scene_builder.py` modülünü yazmak.
  3. `exonym build-scene <candidate_id>` CLI alt komutunu ve `exonym vet` içerisindeki `--auto-scene` bayrağını entegre etmek.

#### 1.2 STScI PRF Şablonu ve Geri-Kazanım Kalibrasyon İndiricisi Eksikliği
* **Hedef Dosya & Satır Aralıkları:**
  - `src/exonym/localization.py:260-320` (`calibrated_prf_assets(workspace: CandidateWorkspace)`)
  - `src/exonym/localization.py:1482-1515` (`run_prf_source_localization` uncalibrated fallback)
  - `schemas/tess-prf-manifest.schema.json`
  - `schemas/tess-prf-recovery-calibration.schema.json`
* **AST ve Hata Yayılım Riski:**
  `calibrated_prf_assets` fonksiyonu `data/external/` altında üç zorunlu kanıt dosyası arar:
  1. `tess_prf.fits` (Resmi STScI PRF 225 KB FITS modeli)
  2. `tess_prf.manifest.json` (`tess-prf-manifest.schema.json` uyumlu, SHA-256 eşleşmeli)
  3. `tess_prf.recovery_calibration.json` (`tess-prf-recovery-calibration.schema.json` uyumlu, `recovery_passed: true`)
  Bu üçlü kanıt zinciri bulunamadığında, `run_prf_source_localization` fonksiyonu satır 1489'da diferansiyel fotometriyi çalıştırmadan doğrudan fail-closed moduna geçer:
  ```json
  {
    "source": "not-run-mission-calibrated-prf-required",
    "calibrated": false,
    "calibration_status": "uncalibrated",
    "validation_eligible": false,
    "conclusion": "inconclusive_mission_calibrated_prf_required"
  }
  ```
  Bu durum `exonym vet` komutunun PRF lokalizasyon kanıtını tanımasını engeller ve hedef yıldız ile komşu nesnelerin diferansiyel piksel ayrıştırmasını kalıcı olarak kilitler.
* **Anayasal İhlal:**
  `AGENTS.md` Kural 4 (Dinamik Enstrüman Modelleri) ve Kural 8. PRF FITS şablonlarının adayın TPF dosyasındaki `CAMERA`, `CCD`, `SECTOR` ve odak düzlemi piksel koordinatlarına göre MAST sunucusundan dinamik çekilmesi ve yerel enjeksiyon-geri-kazanım testinin otomatik icra edilmesi şarttır.

---

### Problem Alanı 2: `/fake-detector` İhlalleri ve Sayısal Provenans Açıkları (Numerical Integrity & Fake Detector Violations)

#### 2.1 Fonksiyonel Taban Kırpmaları (Artificial Floor Clamping / Posterior Choking)
* **Hedef Dosya & Satır Aralıkları:**
  - `src/exonym/vetting/trex/funcs.py:109-110` (`stellar_relations`)
* **Mevcut Hatalı Kod:**
  ```python
  radii[radii < 0.1] = 0.1
  teffs[teffs < 2800.0] = 2800.0
  ```
* **AST ve Fiziksel Etki Analizi:**
  `stellar_relations(masses, max_radii=None, max_teffs=None)` fonksiyonunda $M \le 0.63 M_\odot$ yıldızlar için Chabrier CDWRF spline ızgarası kullanılır. CDWRF ızgara düğümleri (`_Mass_nodes_cdwrf`) $[0.1, 0.63] M_\odot$ aralığında tanımlıdır; en alt düğüm $M = 0.1 M_\odot, T_{\text{eff}} = 2800\text{ K}, R = 0.12 R_\odot$ değerindedir.
  Monte Carlo simülasyonunda $M < 0.1 M_\odot$ (örneğin kahverengi cüceler veya alt-yıldız nesneleri) çekildiğinde spline extrapolasyon üretir. Fonksiyon, bu fiziksel sınır dışı çekimleri reddetmek yerine satır 109-110'da tüm yarıçapları $0.1 R_\odot$'a ve tüm sıcaklıkları $2800\text{ K}$'e yapay olarak sabitler.
* **Hata Yayılım Riski:**
  Monte Carlo çekimlerinde düşük kütleli ikili veya arka plan nesneleri $(0.1 R_\odot, 2800\text{ K})$ noktasında yapay bir tepe (Dirac delta benzeri yoğunlaşma) oluşturur. Bu durum ikincil tutulma derinliklerini, yüzey parlaklığı oranlarını ve seyreltme faktörlerini saptırarak TRICERATOPS tarafından hesaplanan FPP ve NFPP değerlerinde sistematik hata üretir.
* **Anayasal İhlal:**
  `AGENTS.md` Kural 2 ("No Artificial Uncertainty Clipping") ve `/fake-detector` Archetype 2 ("The Posterior Choke"). Fiziksel geçerlilik alanı dışındaki değerler yapay tabanlarla kırpılamaz; fail-closed olmalı (`raise ValueError`) veya Monte Carlo reddetme örneklemesinde geçersiz kabul edilerek elenmelidir.

#### 2.2 Kesilmiş Ondalık Polinom Katsayıları ve Adım Süreksizliği (Precision Destruction & Discontinuity)
* **Hedef Dosya & Satır Aralıkları:**
  - `src/exonym/vetting/trex/priors.py:322-327` ve `338-340` (`lnprior_bound`)
* **Mevcut Hatalı Kod:**
  ```python
  t4 = (alpha * dlogP * (5.5 - 3.4) + f2 * (5.5 - 3.4)
        + (f3 - f2 - alpha * dlogP) * (0.238095 * 5.5 ** 2 - 0.952381 * 5.5 + 0.485714))
  t5 = f3 * (3.33333 - 17.3566 * math.exp(-0.3 * 8.0))
  ...
  f_comp[r34] = t2 + t3 + alpha * dlogP * (m[r34] - 3.4) + f2 * (m[r34] - 3.4) + (
      f3 - f2 - alpha * dlogP) * (0.238095 * m[r34] ** 2 - 0.952381 * m[r34] + 0.485714)
  f_comp[r45] = t2 + t3 + t4 + f3 * (3.33333 - 17.3566 * np.exp(-0.3 * m[r45]))
  ```
* **Derin Matematiksel ve Adli Analiz:**
  1. **Quadratic İntegral Parçası ($\log P \in [3.4, 5.5]$):**
     Moe & Di Stefano (2017, ApJS, 230, 15) Section 4 ikili yıldız periyot integrali:
     $$\int_{3.4}^m \frac{u - 2.0}{2.1} \, du = \left[ \frac{u^2}{4.2} - \frac{2 u}{2.1} \right]_{3.4}^m = \frac{5}{21} m^2 - \frac{20}{21} m - \left( \frac{5 \times 3.4^2 - 68}{21} \right) = \frac{5}{21} m^2 - \frac{20}{21} m + \frac{17}{35}$$
     Burada katsayılar tam rasyonel kesirlerdir:
     $$\frac{5}{21} = 0.2380952380952381..., \quad \frac{20}{21} = 0.9523809523809523..., \quad \frac{17}{35} = 0.4857142857142857...$$
     $m = 5.5$ noktasında analitik değer:
     $$\frac{5}{21}(5.5)^2 - \frac{20}{21}(5.5) + \frac{17}{35} = \frac{49}{20} = 2.45 \text{ (TAM)}$$
     Mevcut kodda yuvarlanmış değerlerle:
     $$0.238095 \times (5.5)^2 - 0.952381 \times 5.5 + 0.485714 = 2.44999225$$
     Burada $\Delta = -7.75 \times 10^{-6}$ sistematik kesme hatası oluşur ve bu hata tüm $m \ge 5.5$ çekimlerine sızar!

  2. **Üstel Kuyruk İntegrali ($\log P \in [5.5, 8.0]$):**
     $$\int_{5.5}^m e^{-0.3 (u - 5.5)} \, du = \left[ -\frac{1}{0.3} e^{-0.3 (u - 5.5)} \right]_{5.5}^m = \frac{10}{3} \left( 1 - e^{-0.3 (m - 5.5)} \right)$$
     Mevcut kod bu ifadeyi $e^{1.65} e^{-0.3 m}$ olarak açmış, $10/3 \approx 3.33333$ ve $(10/3)e^{1.65} \approx 17.3566$ olarak hardcode etmiştir:
     $$m = 5.5 \text{ noktasında: } 3.33333 - 17.3566 \times e^{-1.65} = 3.33333 - 3.3333411 = -1.1 \times 10^{-5} \ne 0$$
     Bu durum $m = 5.5$ sınırında yapay bir adım süreksizliği ($C^0$ ihlali) üretir. Formül `(10.0 / 3.0) * (1.0 - math.exp(-0.3 * (m - 5.5)))` olarak yazıldığında $m = 5.5$ için tam sıfır verir ve süreksizliği yok eder.

* **Anayasal İhlal:**
  `AGENTS.md` Kural 10 ("Precision Preservation & Rational Fractions") ve Kural 13 (Tier 3: Exact Mathematical Forms).

#### 2.3 `sample_q` Kütle Oranı Sınır Hatası ($M_s < 0.1 M_\odot$ Patlaması)
* **Hedef Dosya & Satır Aralıkları:**
  - `src/exonym/vetting/trex/priors.py:216`, `235`
* **Mevcut Kod:**
  ```python
  elif M_s >= 0.3:
      q_min = max(0.1 / M_s, 0.1)
  else:
      q_min = 0.1 / M_s
  ```
* **Adli Teşhis:**
  $M_s < 0.1 M_\odot$ olan düşük kütleli M-cüceleri veya kahverengi cüceler için $q_{\min} = 0.1 / M_s > 1.0$ olmaktadır! Fiziksel ikili kütle oranı tanımı gereği $q \equiv M_{\text{sec}} / M_{\text{prim}} \le 1.0$ olmak zorundadır. $q_{\min} > 1.0$ olduğunda $I_2$ ve $I_3$ integralleri anlamsızlaşmakta ve örnekleme bozulmaktadır.
* **Çözüm:** $M_s$ için alt sınır $M_{\text{min}} = 0.075 M_\odot$ (veya $M_s < 0.1 M_\odot$ için $q_{\min} = \min(0.1 / M_s, 1.0)$) kontrolü eklenmelidir.

#### 2.4 İsimsiz Başlangıç Dispersiyon Vektörü (Inline Anonymous Magic Vectors)
* **Hedef Dosya & Satır Aralıkları:**
  - `src/exonym/transit_fit.py:995-996` (`_fit_emcee_cpu`)
* **Mevcut Hatalı Kod:**
  ```python
  scales = np.full(ndim, 0.01, dtype=float)
  scales[:7] = np.asarray([0.003, 0.03, 0.03, 0.0002, 0.15, 0.03, 0.03])
  ```
* **Anayasal İhlal:**
  `AGENTS.md` Kural 13 (Tier 5 Banned Magic Numbers). `EMCEE_CPU_CORE_PROPOSAL_DISPERSIONS` sabiti tanımlanmalı ve her indeksin fiziksel parametresi belgelenmelidir.

---

### Problem Alanı 3: `AGENTS.md` Kural Şişkinliği ve Biçimsel Kusurlar (Constitutional Bloat & Markdown Defects)

1. **Biçimlendirilmemiş Çıplak Terminal Satırları:**
   `AGENTS.md:52-58` satırları arasında markdown kod bloğu (` ```bash `) olmadan metin arasına serpiştirilmiş çıplak komutlar yer almaktadır.
2. **Aşırı Kural Tekrarı ve Bilişsel Yük (Cognitive Overload):**
   `AGENTS.md:15-20` (Kural 7, 8, 9, 11) ile `AGENTS.md:129` ("Anti-Specification Gaming & Defense Against The Illusion of Working Code") bölümleri aynı maddeleri 4 kez tekrarlamaktadır.
3. **Vetting İddia Kilidi (Claim Unlock) Prosedürünün Belirsizliği:**
   `AGENTS.md:76` satırındaki *"no hand-written claim can unlock the gate until calibrated scene-model integration exists"* ifadesi, sahne modeli ve kalibre PRF entegre edildikten sonra kilit açma sözleşmesini net olarak tanımlamamaktadır.

---

### Problem Alanı 4: Çalışma Ortamı, Venv ve Bağımlılık Senkronizasyonu (Environment Mismatch & Lockfile Parity)

1. **Platform / Python Versiyonu Çelişkisi:**
   - `requirements-lock.txt:1` başlığı: `# EXONYM resolved dependency closure for CPython 3.9 on Windows.`
   - Fiili WSL çalışma ortamı: `Ubuntu Linux 24.04, Python 3.12.3`.
   - `src/exonym/freeze.py:51` içerisindeki `DOCKERFILE_TEMPLATE` ve `APPTAINER_TEMPLATE` `FROM python:3.9-slim` taban imajını zorlamaktadır. Python 3.12 altındaki ABI değişiklikleri ve C-extension paketleri (ör. `blosc2`, `arviz`, `astropy`) için Linux ve WSL üzerinde çalışan deterministik bir kilit profili tescil edilmelidir.
2. **Lightkurve / Oktopus Terminal Uyarı Kirliliği:**
   - `lightkurve` import edilirken `UserWarning: Warning: the tpfmodel submodule is not available without oktopus installed` uyarısı fırlatılmakta ve terminal loglarını kirletmektedir. `pyproject.toml` veya `src/exonym/__init__.py` filtreleriyle bu uyarı susturulmalıdır.

---

### Problem Alanı 5: Dokümantasyon Ağacı Kopuklukları ve Eksik Operatör Kılavuzu (Documentation Misalignment)

1. **Göreceli Bağlantı Hatası:**
   `docs/README.md` Satır 13'te `| ../methods/ | Command-level scientific method records and interpretation limits |` şeklinde üst dizine işaret edilmektedir. Kök dizindeki `methods/`, `protocols/` ve `docs/` arasındaki yapısal ilişki netleştirilmelidir.
2. **Eksik Operatör El Kitabı (Operator Runbook):**
   Yeni nesil bağımsız tarama komutlarının (`survey harvest`, `survey auto-vet`, `survey run-loop`, `survey sensitivity`), sahne oluşturma (`build-scene`) ve PRF kalibrasyonunun (`fetch-prf`) uçtan uca nasıl çalıştırılacağını adım adım gösteren bir operasyon kılavuzu eksiktir.

---

## 3. Atomik Eylem Planı ve Parça Parça TODO Matrisi (Atomic Remediation Matrix & Checklist)

---

### Faz 1: Sayısal Bütünlük ve Adli Provenans (/fake-detector Remediation)

- [ ] **GÖREV 1.1: `funcs.py` İçindeki Yapay Taban Kırpmalarını (`radii < 0.1`, `teffs < 2800.0`) Temizle ve Doğrulama Mekanizması Kur**
  * **Hedef Dosya & Satır Aralıkları:** `src/exonym/vetting/trex/funcs.py:109-111`
  * **İhlal Edilen Kural:** `AGENTS.md` Kural 2 ("No Artificial Uncertainty Clipping") & Kural 13 (Tier 5 Banned Magic Numbers).
  * **Beklenen Girdi/Çıktı Sözleşmesi:**
    - Girdi: `masses: np.ndarray` (yıldız kütleleri $[M_\odot]$).
    - Çıktı: `Tuple[np.ndarray, np.ndarray]` (yarıçaplar $[R_\odot]$ ve etkin sıcaklıklar $[\text{K}]$).
    - Hata Durumu: $M < 0.075 M_\odot$ (hidrojen yakma alt sınırı) durumunda sessizce tabana yapıştırmak yerine `ValueError("Stellar mass out of validated Chabrier CDWRF bounds (<0.075 Msun)")` fırlatılmalı veya vektörel Monte Carlo çağrılarında `np.nan` maskesi üretilmelidir.
  * **Uygulama Spesifikasyonu:**
    1. `src/exonym/vetting/trex/funcs.py` dosyasındaki `radii[radii < 0.1] = 0.1` ve `teffs[teffs < 2800.0] = 2800.0` satırlarını tamamen kaldır.
    2. Düşük kütleli sınır kontrolü ekle:
       ```python
       HYDROGEN_BURNING_MASS_LIMIT_SOLAR: Final[float] = 0.075
       invalid_masses = masses < HYDROGEN_BURNING_MASS_LIMIT_SOLAR
       if np.any(invalid_masses):
           radii[invalid_masses] = np.nan
           teffs[invalid_masses] = np.nan
       ```
    3. TRICERATOPS simülasyon döngüsünün `np.nan` dönen çekimleri doğal olarak reddetmesini sağla.
  * **Hedefli Doğrulama Komutu (WSL):** `pytest tests/test_trex.py -k test_stellar_relations -v`
  * **Kabul Kriteri (Definition of Done):** `funcs.py` içerisinde sıfır boolean taban ataması kalması, testlerin $10^{-12}$ bağıl hassasiyetle yeşil geçmesi.

- [ ] **GÖREV 1.2: `priors.py` İçindeki Polinom Katsayılarını Analitik Rasyonel Kesirlere Dönüştür ve Süreksizliği Gider**
  * **Hedef Dosya & Satır Aralıkları:** `src/exonym/vetting/trex/priors.py:322-327`, `338-340`
  * **İhlal Edilen Kural:** `AGENTS.md` Kural 10 ("Precision Preservation & Rational Fractions") & Kural 13 (Tier 3 Exact Mathematical Forms).
  * **Beklenen Girdi/Çıktı Sözleşmesi:** Moe & Di Stefano (2017) integrasyon katsayıları sembolik kesirlerden türetilmelidir ($5/21$, $20/21$, $17/35$, $10/3$).
  * **Uygulama Spesifikasyonu:**
    1. `src/exonym/vetting/trex/priors.py` modül seviyesinde rasyonel sabitleri tanımla:
       ```python
       import fractions

       # Moe & Di Stefano (2017, ApJS, 230, 15) Section 4 analytical integrals:
       # \int_{3.4}^m ((u - 2.0) / 2.1) du = (5/21) m^2 - (20/21) m + (17/35)
       MOE2017_LOGP_QUAD_C2: Final[float] = float(fractions.Fraction(5, 21))
       MOE2017_LOGP_QUAD_C1: Final[float] = float(fractions.Fraction(20, 21))
       MOE2017_LOGP_QUAD_C0: Final[float] = float(fractions.Fraction(17, 35))

       # Analytical exponential integral: \int_{5.5}^m exp(-0.3 * (u - 5.5)) du = (10/3) * (1 - exp(-0.3 * (m - 5.5)))
       MOE2017_EXP_SCALE: Final[float] = float(fractions.Fraction(10, 3))
       ```
    2. `lnprior_bound` içerisindeki `t4`, `t5` ve `f_comp[r45]` ifadelerini analitik ve $C^0$ sürekli forma dönüştür:
       ```python
       t4 = (alpha * dlogP * (5.5 - 3.4) + f2 * (5.5 - 3.4)
             + (f3 - f2 - alpha * dlogP) * (MOE2017_LOGP_QUAD_C2 * 5.5 ** 2 - MOE2017_LOGP_QUAD_C1 * 5.5 + MOE2017_LOGP_QUAD_C0))
       t5 = f3 * MOE2017_EXP_SCALE * (1.0 - math.exp(-0.3 * (8.0 - 5.5)))
       ...
       f_comp[r45] = t2 + t3 + t4 + f3 * MOE2017_EXP_SCALE * (1.0 - np.exp(-0.3 * (m[r45] - 5.5)))
       ```
  * **Hedefli Doğrulama Komutu (WSL):** `pytest tests/test_trex.py -k test_lnprior_bound -v`
  * **Kabul Kriteri:** Kodda `0.238095`, `0.952381`, `0.485714`, `3.33333`, `17.3566` gibi ham ondalık literallerin kalmaması, $m = 5.5$ noktasında süreksizliğin sıfırlanması, testlerin hatasız geçmesi.

- [ ] **GÖREV 1.3: `priors.py` `sample_q` Kütle Oranı Alt Sınır Patlamasını Düzelt**
  * **Hedef Dosya & Satır Aralıkları:** `src/exonym/vetting/trex/priors.py:216`, `235`
  * **İhlal Edilen Kural:** `AGENTS.md` Kural 13 (Tier 5) & Fiziksel Alan Koruması ($q \le 1.0$).
  * **Uygulama Spesifikasyonu:**
    $M_s < 0.1 M_\odot$ durumunda $q_{\min} = 0.1 / M_s > 1.0$ olmasını engelleyen sınır koruması ekle:
    ```python
    q_min = min(0.1 / max(M_s, 0.075), 0.95)
    ```
  * **Hedefli Doğrulama Komutu (WSL):** `pytest tests/test_trex.py -k test_sample_q -v`
  * **Kabul Kriteri:** $M_s = 0.08 M_\odot$ için `sample_q` çağrıldığında $q > 1.0$ veya negatif integral üretilmemesi.

- [ ] **GÖREV 1.4: `transit_fit.py` Walker Dispersiyon Vektörünü Sabit Olarak İsimlendir ve Belgele**
  * **Hedef Dosya & Satır Aralıkları:** `src/exonym/transit_fit.py:995-996`
  * **İhlal Edilen Kural:** `AGENTS.md` Kural 13 (Tier 5 Banned Magic Numbers).
  * **Beklenen Girdi/Çıktı Sözleşmesi:** MCMC proposal dispersiyon vektörü parametre bazında açık tuple/sözlük olarak tanımlanmalıdır.
  * **Uygulama Spesifikasyonu:**
    1. `src/exonym/transit_fit.py` içerisine modül seviyesinde şu sabiti ekle:
       ```python
       EMCEE_CPU_CORE_PROPOSAL_DISPERSIONS: Final[Tuple[float, ...]] = (
           0.003,   # rp_rstar (planet-to-star radius ratio initial perturbation width)
           0.03,    # log_rho_star (log mean stellar density perturbation width)
           0.03,    # impact_parameter b perturbation width
           0.0002,  # out-of-transit baseline flux perturbation width
           0.15,    # log_jitter photometric white noise perturbation width
           0.03,    # q1 Kipping (2013) triangular limb-darkening perturbation width
           0.03,    # q2 Kipping (2013) triangular limb-darkening perturbation width
       )
       ```
    2. Satır 996'daki `scales[:7] = np.asarray([0.003, 0.03, 0.03, 0.0002, 0.15, 0.03, 0.03])` satırını `scales[:7] = np.asarray(EMCEE_CPU_CORE_PROPOSAL_DISPERSIONS, dtype=float)` ile değiştir.
  * **Hedefli Doğrulama Komutu (WSL):** `pytest tests/test_transit_fit_accelerated.py -v`
  * **Kabul Kriteri:** `scales[:7]` atamasında hiçbir anonim liste kalmaması ve testlerin yeşil geçmesi.

---

### Faz 2: Eksik Otomasyon Köprülerinin İnşası (Scene & PRF Pipeline Builders)

- [ ] **GÖREV 2.1: `trex-scene-manifest.schema.json` Şemasını Tanımla ve Tescil Et**
  * **Hedef Dosya(lar):**
    - `schemas/trex-scene-manifest.schema.json` [YENİ]
    - `src/exonym/_resources/schemas/trex-scene-manifest.schema.json` [YENİ]
  * **İhlal Edilen Kural:** Wheel-fallback kaynak paritesi kuralı (`AGENTS.md` Layout And Ownership).
  * **Girdi/Çıktı Sözleşmesi:** Draft 2020-12 JSON Schema; `schema_version: 1`, `candidate_id`, `source: "candidate-data"`, `target`, `archival_gaia`, `contrast_curve`, `background`, `resolved_neighbors` alanlarını zorunlu kılar.
  * **Şema Spesifikasyonu:**
    ```json
    {
      "$schema": "https://json-schema.org/draft/2020-12/schema",
      "$id": "https://example.invalid/schemas/trex-scene-manifest-1.0.json",
      "title": "Candidate-local TREX scene manifest",
      "type": "object",
      "additionalProperties": false,
      "required": ["schema_version", "candidate_id", "source", "target", "archival_gaia", "contrast_curve", "background", "resolved_neighbors"],
      "properties": {
        "schema_version": {"const": 1},
        "candidate_id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9._-]*$"},
        "source": {"const": "candidate-data"},
        "target": {
          "type": "object",
          "additionalProperties": false,
          "required": ["ra_deg", "dec_deg", "mass_solar", "radius_solar", "teff_k", "parallax_mas", "tess_mag"],
          "properties": {
            "ra_deg": {"type": "number", "minimum": 0.0, "maximum": 360.0},
            "dec_deg": {"type": "number", "minimum": -90.0, "maximum": 90.0},
            "mass_solar": {"type": "number", "exclusiveMinimum": 0.0},
            "radius_solar": {"type": "number", "exclusiveMinimum": 0.0},
            "teff_k": {"type": "number", "exclusiveMinimum": 0.0},
            "parallax_mas": {"type": "number", "exclusiveMinimum": 0.0},
            "tess_mag": {"type": "number"}
          }
        },
        "archival_gaia": {
          "type": "object",
          "additionalProperties": false,
          "required": ["path", "sha256", "target_source_id", "neighbor_source_ids"],
          "properties": {
            "path": {"const": "outputs/archival_vetting_report.json"},
            "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "target_source_id": {"type": "string", "minLength": 1},
            "neighbor_source_ids": {"type": "array", "items": {"type": "string", "minLength": 1}}
          }
        },
        "contrast_curve": {
          "type": "object",
          "additionalProperties": false,
          "required": ["path", "sha256", "separations_arcsec", "delta_magnitudes"],
          "properties": {
            "path": {"type": "string", "pattern": "^data/external/.*$"},
            "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "separations_arcsec": {"type": "array", "minItems": 2, "items": {"type": "number", "minimum": 0.0}},
            "delta_magnitudes": {"type": "array", "minItems": 2, "items": {"type": "number", "minimum": 0.0}}
          }
        },
        "background": {
          "type": "object",
          "additionalProperties": false,
          "required": ["path", "sha256", "model", "star_count"],
          "properties": {
            "path": {"type": "string", "pattern": "^data/external/.*$"},
            "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "model": {"enum": ["trilegal", "background"]},
            "star_count": {"type": "integer", "minimum": 0}
          }
        },
        "resolved_neighbors": {
          "type": "array",
          "items": {
            "type": "object",
            "additionalProperties": false,
            "required": ["source_id", "mass_solar", "radius_solar", "delta_mag", "separation_arcsec"],
            "properties": {
              "source_id": {"type": "string", "minLength": 1},
              "mass_solar": {"type": "number", "exclusiveMinimum": 0.0},
              "radius_solar": {"type": "number", "exclusiveMinimum": 0.0},
              "delta_mag": {"type": "number"},
              "separation_arcsec": {"type": "number", "exclusiveMinimum": 0.0}
            }
          }
        }
      }
    }
    ```
  * **Hedefli Doğrulama Komutu (WSL):** `pytest tests/test_schemas.py -v`
  * **Kabul Kriteri:** Şema doğrulama testlerinin ve kaynak-parite testinin tam geçmesi.

- [ ] **GÖREV 2.2: Otonom TREX Sahne Oluşturucu Motorunu (`scene_builder.py`) Yaz**
  * **Hedef Dosya:** `src/exonym/vetting/trex/scene_builder.py` [YENİ]
  * **İhlal Edilen Kural:** `AGENTS.md` Kural 8 ("Prohibition on Hollow 'Fail-Closed' Walls").
  * **Girdi Sözleşmesi:**
    - `workspace: CandidateWorkspace`
    - `outputs/archival_vetting_report.json` (Gaia DR3 hedef ve tüm komşu yıldızları).
    - `data/external/stellar_params.json` veya TIC v8.2 katalog yan ürünü.
    - Varsa `data/external/*contrast*` eğrisi; yoksa enstrümantal sınır eğrisi oluşturulmalı.
  * **Çıktı Sözleşmesi:**
    - `data/external/trex_scene.json` (SHA-256 mühürlü, şema-uyumlu).
    - Return type: `Path` (oluşturulan dosyanın tam yolu).
  * **Uygulama Spesifikasyonu:**
    1. Arşiv raporundan hedef koordinatlarını ve tüm komşuları oku (asla `[:5]` kesme yapma!).
    2. Hedef yıldız parametrelerini ($M_*, R_*, T_{\text{eff}}, \varpi, T_{\text{mag}}$) doğrula.
    3. Komşular için $\Delta\text{mag} = \text{mag}_{\text{neighbor}} - \text{mag}_{\text{target}}$ hesapla, Torres/Chabrier modellerinden $M_*$ ve $R_*$ türet.
    4. Kontrast eğrisini ve TRILEGAL arka plan popülasyonunu adayın galaktik koordinatlarına göre bağla.
    5. Dosyayı atomik olarak yaz (`_write_json_atomic`), SHA-256 özetini üret.
  * **Hedefli Doğrulama Komutu (WSL):** `pytest tests/test_trex.py -k test_scene -v`
  * **Kabul Kriteri:** Aday çalışma alanında `trex_scene.json` dosyasının şemaya tam uyumlu üretilmesi.

- [ ] **GÖREV 2.3: `exonym build-scene <candidate_id>` CLI Komutunu Entegre Et**
  * **Hedef Dosya:** `src/exonym/__main__.py`
  * **Uygulama Spesifikasyonu:**
    1. CLI alt komutu ekle: `exonym build-scene <candidate_id> [--overwrite]`.
    2. `exonym vet <candidate_id>` komutuna `--auto-scene` bayrağı ekle; dosya yoksa ve arşiv raporu mevcutsa sahneyi otomatik oluştursun.
  * **Hedefli Doğrulama Komutu (WSL):** `pytest tests/test_vetting.py -k test_cli -v`
  * **Kabul Kriteri:** Komut çalıştırıldığında exit code 0 ile `trex_scene.json` dosyasının oluşması.

- [ ] **GÖREV 2.4: STScI MAST TESS PRF İndirici Modülünü (`fetch_mission_calibrated_prf`) Yaz**
  * **Hedef Dosya:** `src/exonym/localization.py`
  * **İhlal Edilen Kural:** `AGENTS.md` Kural 4 & 5 (Dinamik Enstrüman Modelleri).
  * **Girdi Sözleşmesi:**
    - Adayın TPF dosyası (`data/raw/tess*tp.fits`).
    - FITS başlık kartları: `CAMERA`, `CCD`, `SECTOR`, `CRPIX1`, `CRPIX2`.
  * **Çıktı Sözleşmesi:**
    - `data/external/tess_prf.fits` (MAST HTTP sunucusundan indirilen resmi PRF FITS).
    - `data/external/tess_prf.manifest.json` (`tess-prf-manifest.schema.json` uyumlu).
  * **Uygulama Spesifikasyonu:**
    1. TPF başlığından Camera, CCD ve odak düzlemi piksel koordinatlarını dinamik oku.
    2. STScI MAST resmi URL'sinden (`https://archive.stsci.edu/missions/tess/models/prf_fits_files/...`) en yakın PRF ızgara noktasını çek.
    3. İndirilen FITS dosyasını atomik olarak kaydet, SHA-256 özetini çıkar ve manifesti yaz.
  * **Hedefli Doğrulama Komutu (WSL):** `pytest tests/test_localization.py -k test_fetch_prf -v`
  * **Kabul Kriteri:** Ağ bağlantısıyla veya yerel önbellekten resmi FITS'in inmesi ve manifestin şemayı geçmesi.

- [ ] **GÖREV 2.5: PRF Geri-Kazanım Kalibrasyon Motorunu (`calibrate_prf_recovery`) Yaz**
  * **Hedef Dosya:** `src/exonym/localization.py`
  * **Girdi Sözleşmesi:**
    - `data/external/tess_prf.fits`
    - Adayın medyan TPF görüntüsü.
  * **Çıktı Sözleşmesi:**
    - `data/external/tess_prf.recovery_calibration.json` (`tess-prf-recovery-calibration.schema.json` uyumlu, `recovery_passed: true`).
  * **Uygulama Spesifikasyonu:**
    1. Resmi PRF modelini kullanarak TPF medyan görüntüsüne bilinen akıda sahte nokta kaynak enjekte et.
    2. PRF fotometrisiyle kaynağın akısını ve konumunu geri kazan.
    3. Geri kazanım hatası sınır içinde kaldığında `recovery_passed: true` bayrağını oluştur ve imzalı manifesti yaz.
  * **Hedefli Doğrulama Komutu (WSL):** `pytest tests/test_localization.py -k test_calibrated_prf_assets -v`
  * **Kabul Kriteri:** `calibrated_prf_assets(workspace)` fonksiyonunun `(assets, None)` dönmesi ve `calibrated: true` durumuna geçilmesi.

- [ ] **GÖREV 2.6: `exonym localization fetch-prf <candidate_id>` CLI Komutunu Entegre Et**
  * **Hedef Dosya:** `src/exonym/__main__.py`
  * **Uygulama Spesifikasyonu:** `localization` komut grubuna `fetch-prf` alt komutunu ekle; adayın raw TPF'lerinden otomatik algılayıp indirme ve kalibrasyonu tek adımda yapsın.
  * **Hedefli Doğrulama Komutu (WSL):** `pytest tests/test_localization.py -v`
  * **Kabul Kriteri:** CLI üzerinden PRF varlıklarının üretilmesi ve ardından `exonym localization` komutunun `calibrated: true` raporu üretmesi.

---

### Faz 3: Anayasal Temizlik ve `AGENTS.md` Senkronizasyonu (Constitutional Hygiene)

- [ ] **GÖREV 3.1: Çıplak Terminal Komutlarını (Lines 52-58) Biçimlendir**
  * **Hedef Dosya & Satır Aralıkları:** `AGENTS.md:52-58`
  * **Uygulama Spesifikasyonu:** Çıplak satırları fenced markdown code-block (` ```bash `) içerisine al, test çalıştırma açıklamalarıyla bütünleştir.
  * **Hedefli Doğrulama Komutu (WSL):** `exonym lint-paths --source`
  * **Kabul Kriteri:** Markdown linter uyarılarının sıfırlanması.

- [ ] **GÖREV 3.2: Mükerrer Hile Karşıtı Maddeleri Tek Bir Yetkili Bölümde Birleştir**
  * **Hedef Dosya & Satır Aralıkları:** `AGENTS.md:15-20` ve `AGENTS.md:129`
  * **Uygulama Spesifikasyonu:** Kural 7, 8, 9, 11 ile "Anti-Specification Gaming" paragraflarını tek bir yetkili "Adli Bilimsel Dürüstlük ve Hile Yasağı (Forensic Scientific Integrity Directive)" başlığı altında birleştir; anlam kaybı olmadan mükerrer cümleleri buda.
  * **Kabul Kriteri:** `AGENTS.md` dosyasının net, okunabilir ve şişkinlikten arınmış hale gelmesi.

- [ ] **GÖREV 3.3: Bilimsel Vetting İddia Kilidi Açma (Claim Unlock) Kriterlerini Netleştir**
  * **Hedef Dosya & Satır Aralıkları:** `AGENTS.md:76-80`
  * **Uygulama Spesifikasyonu:** "no hand-written claim can unlock the gate" ifadesini güncelle; TREX sahne modeli (`trex_scene.json`) ve kalibre STScI PRF entegre edildiğinde doğrulanmış adayın hangi nicel kriterlerle (`FPP < 0.01`, `NFPP < 0.001`, odd-even $|Z| < 3$, target-to-other centroid ratio $> 1.0$) onaylanabilir hale geleceğini açıkça belgele.
  * **Kabul Kriteri:** Anayasal kılavuz ile fiili bilimsel boru hattı sözleşmesinin tam örtüşmesi.

---

### Faz 4: Çalışma Ortamı, Kilit Dosyaları ve Bağımlılıklar (Environment & Lockfile Parity)

- [ ] **GÖREV 4.1: WSL Ubuntu Python 3.12 Kilit Dosyasını Güncelle ve Belgele**
  * **Hedef Dosya:** `requirements-lock.txt`
  * **Uygulama Spesifikasyonu:** Dosya başlığındaki "CPython 3.9 on Windows" ibaresini güncelle. CI (Python 3.9) ve WSL (Python 3.12) çift-çalışma zamanı destek sözleşmesini açıkla ve paket sürümlerini Linux x86_64 ortamıyla uyumlu hale getir.
  * **Kabul Kriteri:** `freeze.py` ve `verify-release` testlerinin WSL Python 3.12 ortamında temiz çalışması.

- [ ] **GÖREV 4.2: `freeze.py` İçindeki Dockerfile/Apptainer Şablonlarını Dinamik Hale Getir**
  * **Hedef Dosya & Satır Aralıkları:** `src/exonym/freeze.py:51`, `64`
  * **Uygulama Spesifikasyonu:** Sabit `python:3.9-slim` ifadesini aktif Python sürümüne veya parametrik sürüme bağla (`python:{0}.{1}-slim`).
  * **Hedefli Doğrulama Komutu (WSL):** `pytest tests/test_freeze.py -v`
  * **Kabul Kriteri:** Release freeze testlerinin hatasız geçmesi.

- [ ] **GÖREV 4.3: `oktopus` / Lightkurve Terminal Uyarısını Sustur**
  * **Hedef Dosya:** `pyproject.toml` veya `src/exonym/__init__.py`
  * **Uygulama Spesifikasyonu:**
    ```python
    import warnings
    warnings.filterwarnings("ignore", message=r".*the tpfmodel submodule is not available without oktopus installed.*")
    ```
  * **Hedefli Doğrulama Komutu (WSL):** `exonym --help`
  * **Kabul Kriteri:** CLI çalışırken stderr üzerinde hiçbir üçüncü parti kütüphane uyarısının belirmemesi.

---

### Faz 5: Dokümantasyon Hizalaması ve Operatör El Kitabı (Documentation Alignment & Operator Runbook)

- [ ] **GÖREV 5.1: `docs/README.md` İçindeki Göreceli Dizin Bağlantılarını Düzelt**
  * **Hedef Dosya:** `docs/README.md`
  * **Uygulama Spesifikasyonu:** `../methods/` ve `../protocols/` gibi göreceli bağlantıları kök dizin hiyerarşisiyle tutarlı hale getir, her dizinin sorumluluk alanını netleştir.
  * **Hedefli Doğrulama Komutu (WSL):** `exonym lint-paths --source`
  * **Kabul Kriteri:** Sıfır linter kırılması, dokümantasyon ağacının tam tutarlılığı.

- [ ] **GÖREV 5.2: Uçtan Uca "Autonomous Operator Runbook" Rehberini Hazırla**
  * **Hedef Dosya:** `docs/EXONYM_SCIENTIFIC_ARCHITECTURE_AND_HOW_TO.md`
  * **Uygulama Spesifikasyonu:** Yeni eklenen `survey harvest`, `survey auto-vet`, `survey run-loop`, `build-scene` ve `fetch-prf` komutlarını içeren 7 Fazlı operasyon zincirini (`intake -> feasibility -> acquisition -> vetting -> followup -> analysis -> review`) adım adım kod bloklarıyla belgele:
    ```bash
    # FAZ 1: Intake & Tescil
    exonym init-candidate <id> --tic <TIC_ID>
    exonym set-state <id> --state intake --reason "Candidate ingested from survey alert"

    # FAZ 2: Feasibility & Arşiv Sorgusu
    exonym archive query <id>
    exonym survey harvest <survey_id> <id>
    exonym advance <id> --phase feasibility

    # FAZ 3: Acquisition & FITS Doğrulama
    exonym download <id> --all-sectors
    exonym advance <id> --phase acquisition

    # FAZ 4: Vetting & Otomasyon Köprüleri (Scene + PRF)
    exonym detrend <id> --method wotan
    exonym search <id> --method bls
    exonym build-scene <id>
    exonym localization fetch-prf <id>
    exonym localization <id>
    exonym vet <id> --auto-scene
    exonym advance <id> --phase vetting

    # FAZ 5: Followup & Karakterizasyon
    exonym fit <id> --sampler emcee
    exonym ttv <id>
    exonym phasecurve <id>
    exonym advance <id> --phase followup

    # FAZ 6: Review & Yayın Hazırlığı
    exonym export-paper <id>
    exonym advance <id> --phase review
    ```
  * **Kabul Kriteri:** Yeni bir operatör veya otonom ajanın kılavuzu takip ederek sıfırdan adayı başarıyla doğrulayabilmesi.

---

## 4. Adli Doğrulama Kapısı ve Kabul Kriterleri (Forensic Verification Gate & Exit Criteria)

Tüm adımlar tamamlandığında sistem WSL ortamında (`Ubuntu Linux, Python 3.12`) şu komut zinciri ile mühürlenmelidir:

```bash
# ==============================================================================
# 1. Bilimsel ve Adli Birim Testleri (Targeted Scientific & Forensic Pytest)
# ==============================================================================
pytest tests/test_trex.py tests/test_localization.py tests/test_vetting.py tests/test_transit_fit_accelerated.py -v

# ==============================================================================
# 2. Mimari, Kapı ve Bağımsız Tarama Testleri (Architectural Gates & Discovery)
# ==============================================================================
pytest tests/test_gates.py tests/test_survey.py tests/test_discovery.py -q

# ==============================================================================
# 3. Statik Derleme ve Güvenlik Denetimi (Bytecode Compilation & Security Audit)
# ==============================================================================
python -m compileall -q src tests
bandit -r src -lll

# ==============================================================================
# 4. İdari Dizin Düzeni ve Şema Denetimi (Administrative Path & Schema Linter)
# ==============================================================================
exonym lint-paths --source
```

### Kusursuzluk İmzası (Definition of Architectural Perfection):
* **Sıfır Yapay Taban:** `src/` altında hiçbir `[ < 0.1] = 0.1` veya `np.clip` posterior boğucu kalmamıştır.
* **Tam Rasyonel Kesirler ve $C^0$ Süreklilik:** Moe & Di Stefano integralleri IEEE 754 çift duyarlıklı rasyonel kesirlerle ($5/21, 20/21, 17/35$) ifade edilmiş, $\log P = 5.5$ süreksizliği giderilmiştir.
* **Fiziksel Kütle Oranı Alanı ($q \le 1.0$):** $M_s < 0.1 M_\odot$ için patlayan $q_{\min} = 0.1 / M_s$ hatası giderilmiştir.
* **Kesintisiz Otomasyon:** `build-scene` ve `fetch-prf` komutları sayesinde TRICERATOPS ve PRF lokalizasyonu manuel müdahaleye gerek kalmadan otonom olarak `calibrated: true` raporu üretmektedir.
* **Jilet Gibi Anayasa:** `AGENTS.md` biçimsel hatalardan ve mükerrer kurallardan arındırılmış, yürütme ortamı ile %100 uyumlu hale getirilmiştir.
