# ReuseAcrossPersistent (RAP) — 重建與 v0 完成計畫

## 0. 這份計畫的來源

RAP v0 曾經在 2026-08-14 ~ 08-15 完整實作並驗證通過，但因為成果從未離開工作目錄（`git reflog` 只有 `clone` 與 `checkout` 兩筆、無 stash、無 commit），伺服器維修後全部遺失。目前 `public/rocm-libraries` 是 2026-08-17 14:42 的全新 clone，基底 commit `b1df07af887`，分支 `users/joschang/reuse_across_persistent`，工作樹乾淨。

唯一倖存的是那次工作的完整對話紀錄 `/home/josh/cursor_tensilelite_rap_kernel_plan.md`（5572 行）。這份計畫是從那份紀錄還原出來的實作食譜，加上針對**新機器環境**的修正，以及 v0 未完成的部分（解開 K 的上限）。

**這份計畫的目標比上次多一段**：上次結束時 K 被結構性地鎖在 `(PGR+1) × DepthU = 768`，本計畫把那條限制也解開。

---

## 0.5 現況與修正（2026-08-18 補記）

M1 與 M2 都已完成，壓縮成單一 commit `8fd7413240d` 並推上 `origin/users/joschang/reuse_across_persistent`。

**這一節的優先權高於本文其餘部分。** 計畫是實作**之前**寫的，實作過程量到的事實推翻了其中幾處。原文保留不動，因為它記錄了設計推理的來由；但讀到本節指名修正的段落時，以本節為準。

**1. 步驟 7 混淆了 `k` 與 `k_max`。**（修正 §5 步驟 7）`k` 不是可以挑的旋鈕。predicate 是 `SizeEqual(K) = k × DepthU`（`Contractions.py:604-619`），所以 `k` 恆等於 `K / DepthU`，**一個 solution 只服務一個 K**。而推導階段結構上看不到 problem size（`assignDerivedParameters` 的簽章裡沒有），所以 `k` 也不可能「從 K 算出來」。步驟 7 寫的「predicate 由 `k_max` 發」照字面實作的後果是：kernel 只服務 `k_max × DepthU` 這一個尺寸。v0 接受這個後果，代價是同一個 config 無法覆蓋多個 K。放寬需要讓 NGLL/NLL 不再綁死在 k 範圍的頂端（`KernelWriter.py:4099` 的 `numKTiles - 1 - remainPgr`），留給 v1。

**2. `k_max` 的估計值錯了。**（修正 §2.2 表格與 §5 步驟 7 的驗收）實測方式是掃 `TENSILE_RAP_KTILES` 直到 store 守門觸發：MT64x512 是 **3**（與估計相符，且正好等於 `PGR+1`，餘裕為零），MT64x256 是 **8**（估計為 5）。所以 MT64x256 服務 K=2048、MT64x512 服務 K=768。

**3. `k` 還有第二條上限，計畫完全沒提到。** `errorCode=1`（`too many vgprs`）：MT64x512 在 k=7、MT64x256 在 k≈11 就撞上暫存器總量，那時 store 守門根本沒機會跑。兩個 config 都是 store 界先咬到，所以只模型 store 那一界剛好夠用——但那是運氣，不是設計。

**4. §2.3 B 的結論對，理由錯。** 「codegen 的值到不了 predicate」在**同一次 build 內**成立，但出貨流程根本不重算推導：logic yaml 的每個 solution 都帶 `AssignedDerivedParameters: true`，而 `Solution.py:1672-1674` 看到它就直接 return，位置在 RAP 推導區塊之前。所以出貨時用的是 logic yaml 記錄的 `_RAPNumResidentKTiles`，模型只在 tuning 流程跑。這讓模型估錯的風險比計畫假設的小得多——tuning 會真的建置、真的過守門、真的驗證。

**5.「被守門毒化的 solution 仍會執行」是 `ForceGenerateKernel` 造成的，不是缺少機制。**（修正 §7 風險表）移除機制本來就在：`kernelBody` 回傳的 error 就是 `overflowedResources`（`KernelWriter.py:6981`），`_getKernelSource` 在 error≠0 時 raise（`:11535`），`getSourceFileString` 接住給 `errcode = -2`，`removeInvalidSolutionsAndKernels` 就把 solution 移除。yaml 已停用該旗標並註明原因。停用是必要的，不是整潔問題：`LibraryLogic` 只依 gflops 排名、不讀 validation 欄，而 `s_endpgm` 的 stub 快得離譜，會贏得 benchmark 並被記錄進 logic yaml。

**6. 驗收條件變了。**（修正 §3.1 與 §3.3 的「四個 kernel 全部 PASSED」）兩個 config 服務不同的 K，所以單一尺寸不可能讓四個 kernel 都 PASSED。yaml 同時放 K=768 與 K=2048 時，每個 RAP1 kernel 在自己的尺寸 PASSED、在另一個回報 `DID_NOT_SATISFY_ASSERTS`。組語結構比對與完整 unit test 的要求不變。

**7. 步驟 8 擔心的 LDS 雙緩衝 parity 問題沒有發生。**（修正 §5 步驟 8 的「一個要驗證的假設」）k=3 是 1 份 body（奇數）、k=8 是 6 份（偶數），兩者都 PASSED 且組語結構比對通過。emitter 循序推進 token 的推論是對的。

**8. 模型唯一的經驗常數是 `S = 40`。** 它吸收了「store 當下非 ValuC、非常駐的活躍暫存器」（量到 24 / 28）、withheld 區塊內的 2 個對齊填充、以及該區塊借回給 store 時的碎片損失（量到 14 / 10）——兩個 config 的合計都恰好是 40。能同時重現兩個實測上限的範圍是 29..52，所以安全預算是兩側各約 12 個暫存器，不到 `R` 的五分之一。這個數字與兩個量測點都寫進了 `rapMaxResidentKTiles` 的註解與 unit test，所以上游暫存器配置若改動使它失效，會是測試先失敗，而不是守門默默丟掉 solution。

**9. 步驟 4b 的「約 240 個 `Label(...)` 站點」估計錯了，label 後綴不需要獨立模組。** 曾經有一個 `Tensile/RapLabel.py`，用 shadow `Label` 這個名字的方式讓後綴自動套用，已在 2026-08-18 移除。真正會撞名的只有 **12 個** label 名稱（`LoopBeginL`、`LoopEndL`、`openLoopL`、`PrefetchGlobalLastIterEnd`、`skipPGR2_1/_2`、`SkipStaggerA/B/MXSA/MXSB`、`toPGR1`、`toPGR1end_OrdNLL`），因為其餘的走 `labels.getNameInc` 的計數器、跨兩次發射持續遞增、本來就唯一。現在的機制是 `KernelWriter` 上的 `rapLabel()` / `rapGetName()` 兩個方法加 `rapIterNLabels()` context manager，在 30 個站點顯式呼叫。這樣做的四個好處：不依賴 import 順序（shim 版本靠 `KernelWriterAssembly` 的 `from .KernelWriterModules import *` 恰好 re-export 同名物件才生效，一旦有人整理那個 import 就會**安靜地**失效並產生 duplicate label abort）；不需要手維護 `SHARED_LABELS` 例外清單（極性反過來了：只有顯式要求的才套後綴，區段外的 label 沒有東西可以寫錯）；不需要 subclass nanobind 型別，連帶消掉「isinstance 要用 `RocisaLabel`」那條註記；在使用處是明示的。

換掉機制之後，加後綴的 label 從 53 個降到 13 個，因此 basic block 邊界改變，後端重排了一小段指令。淨差異是 MT64x512 少一個 `s_wait_dscnt`、多一個 `s_nop`，MT64x256 另外少 13 個 `s_set_vgpr_msb`。這兩類差異都查證過：被移動的兩個 WMMA 的 wait 立即數各自加上中間插入的 `ds_load` 筆數（4+3=7、3+9=12），保證的那批較舊 load 不變；被刪掉的 `s_wait_dscnt 35` 緊接在更強的 `s_wait_dscnt 31` 之前、且兩個 WMMA 讀同一組暫存器，屬於嚴格冗餘。`s_set_vgpr_msb` 那類由驗證覆蓋（bank 選錯會讀到錯的實體暫存器，功能模擬器抓得到）。

參照組（各檢查點的 `.s` 與 patch）在 `/home/josh/rap_refs/`，`k_max` 模型那一步是 `step9_kmax_model/`——注意它是**換掉 label 機制之前**產出的，所以與現在的組語有上述差異。

---

## 1. 環境與基準

| 項目 | 值 |
|---|---|
| repo | `/home/josh/public/rocm-libraries`（容器內為 `/josh/public/rocm-libraries`） |
| 分支 / 基底 | `users/joschang/reuse_across_persistent` @ `b1df07af887` |
| 執行環境 | docker `rocm/hipblaslt-private:latest`，掛載 `/home/josh:/josh` |
| GPU | **無實體 GPU**。FFM 功能模擬器（topology `mi450`），target `gfx1250`，回報 32 CU |
| 驗證指令 | `./build_tmp/Tensile.sh --rocm-agent-enumerator=rocm_agent_enumerator mxf8mxf4_gfx1250_rap.yaml mxf8mxf4_gfx1250_rap` |

**已知的環境障礙**：repo 由 root 擁有（在容器內 clone），主機上的一般使用者無法寫入。動工前需要 `sudo chown -R josh:josh /home/josh/public/rocm-libraries`，或全程在容器內以 root 操作。

**基準已建立**：2026-08-17 15:27 的 run，MT64x256 的 RAP=0 kernel 對 `(64, 32768, 4, 768)` 回報 `PASSED`。所以環境是好的，可以直接接續。

### 測試 problem 的性質

`- Exact: [64, 32768, 4, 768]`，索引順序是 `M, N, batch, K`。

- `M = 64 = MacroTile0` → `NumWorkGroups0 == 1`，A 不隨 `WorkGroup0` 改變
- `N = 32768`，`MT1 = 256` → 128 個 N-tile；`MT1 = 512` → 64 個
- `batch = 4` → 總 tile 數 512（MT64x256）或 256（MT64x512）
- 32 個 WG → 每個 WG 跑 16（或 8）個 tile，確實會跨多個 tile，RAP 的改動測得到
- `K = 768 = 3 × DepthU`

**注意一個上次沒有的性質**：DP-only 下每個 WG 走的 tile 是 `flat += skGrid`（`Components/StreamK.py` 的 `StreamKIter += skGrid × ItersPerTile`），不是連續的一段。若 `skGrid == 32`，WG0 走 tile 0, 32, 64, …，而 batch 邊界落在 tile 128 / 256 / 384——**每個 WG 都會跨過 batch 邊界**。上次幾次 batch 測試都因為 tile 數剛好對齊而沒有測到這件事，這次天然會測到。動工時要用實際的 kernarg 把 `skGrid` 確認一次。

---

## 2. 決定計畫形狀的關鍵事實

### 2.1 這台機器改變的事

| 事實 | 後果 |
|---|---|
| FFM 是**功能**模擬器，不模擬時序 | 缺 waitcnt、缺 barrier **不會**讓驗證失敗。上次六個 bug 裡有兩個是這類。**「PASSED」從此不等於「正確」** |
| 沒有可信的效能數字（kernel time 4.27e6 µs） | 上次的步驟 0（上限實驗，量出 5.7% / 10.1% 天花板）與步驟 4 的「±2% 才算通過」全部失效，本計畫不做效能量測 |
| 上次挑 config 的理由（MT64x256 量測散布 0.09%）不再成立 | 改用別的理由挑 config：偵錯訊號的獨立性、store 餘裕 |

### 2.2 上次確立、這次仍然成立的事實

| 事實 | 後果 |
|---|---|
| `ReuseAcrossPersistent` 在 codebase 完全不存在 | 第一件事是參數接線，否則 yaml 連 parse 都不會過 |
| 現有 kernel `.amdhsa_next_free_vgpr` 逼近上限，但那是 **epilogue** 高水位，主迴圈只到約 613 | RAP 放得下，代價是 store 的暫存器預算變小 |
| store 批次 = `numVgprAvailable // numVgprsPerElement`，每 thread 元素數 `E = ValuC / StoreVectorWidth` | store 成本是**階梯**不是斜坡。MT64x512 的 `E = 128`、MT64x256 的 `E = 64` |
| 每多常駐一個 k-tile 吃掉 **68** 個 VGPR（ValuA 64 + MXSA 4） | 兩個 config 的 `k_max` 不同：MT64x512 約 3、MT64x256 約 5 |
| MX scale 運算元在 `s_set_vgpr_msb` 沒有欄位（該指令只有 src0/src1/src2/dst 四個 2-bit 欄位，scaled WMMA 有六個運算元） | MXSA / MXSB **必須**待在 v0–v255。這是編碼限制，不是 bank 偏好。現有 layout 本來就把 MX scale 放在 v0 |
| K 迴圈的 trip count 是 runtime SGPR，repo 裡沒有完全展開的機制 | 但 `ItersPerTile − PGR = 1` 時迴圈剛好只跑一次，所以 k=3 不需要展開機制。k>3 才需要 |
| `s_wait_dscnt` 立即數由 `states.numReadsPerIter*` 常數解析算出，與指令流脫鉤 | 算多了 wait 變 no-op → 偶發 `-nan`。但見 2.3 A 的修正 |
| StreamK 的 tile 空間跨 batch 線性化 | batch 是真實的正確性風險；見 §7 的應變 |
| yaml 寫 `AssertSummationElementMultiple: 256`，產出的 solution 卻是 32（kernel 名稱可見 `ASEM32`） | 既有的鬆動，**不在本計畫範圍內修**。但要知道它存在，因為新加的 `SizeEqual(K)` 會是唯一真正鎖住 K 的東西 |

### 2.3 這次新查到、會改變設計的事實

**A. 大部分 waitcnt 不是 Python 發的。** `ScheduleIterAlg=4` 會被改寫成 `_ScheduleIterAlg=0` + `_StinkyTofuOptLevel=3`，iterN 那 52 個 `s_wait_dscnt` 裡只有 7 個來自 Python IR，其餘由 StinkyTofu 後端的 `StinkyWaitCntInsertionPass` 依真實資料流插入。這有兩個推論：

- 上次擔心的 `numReadsPerIter*` 低估問題，在這個 config 下影響有限（後端會重算）
- 但也代表 **codegen 時期的檢查抓不到後端造成的差異**。上次那個「iterN 一個 waitcnt 都沒有」的 bug，Python 端兩次都正確呼叫了 `_wait` 14 次——只有產完組語之後的比對看得到它

**B. codegen 階段算出來的值無法餵給出貨流程的 predicate。** benchmark flow 的順序是「推導 → codegen → 發 predicate → 寫 YAML」（有利），但出貨用的 `TensileCreateLibrary` flow 是相反的：predicate 在 `TensileCreateLibrary/Run.py:1087` 就從 logic YAML 建好，codegen 要到 `:1101` 才跑。另外 codegen 跑在 joblib 的 worker **行程**裡，直接 `kernel[...] = ` 的寫入會被丟棄——但 `CpuThreads: 1` 時走的是父行程的 list comprehension，**會看起來正常運作**然後在平行建置下安靜地壞掉。

現成的回寫管道確實存在（`CUOccupancy` 走 `KernelCodeGenResult` → `passPostKernelInfoToSolution`，`Run.py:298-310`），但它落在 `sizeMapping` 而不是 predicate，而且只在 benchmark flow。**codebase 裡沒有任何「codegen 階段的值餵給 predicate」的前例。**

→ 所以 `k_max` 必須在**推導階段**用解析模型算出來，codegen 的 store 守門只負責**驗證**。

**C. 完全展開 unroll loop 會讓 `InitCIterWmma` 安靜地失效。** `RegionClonePass`（`shared/stinkytofu/src/transforms/asm/RegionClonePass.cpp`）用字面標籤 `label_LoopBeginL` 找區域，並用「跳回同一標籤的回邊」當掃描邊界。拿掉任何一個，`findRegions` 回傳空集合、pass 直接 `continue` **不報錯**；而 Python 端在主迴圈路徑上已經把 `v_mov` 歸零 C 的程式碼跳掉了（`KernelWriterAssembly.py:6243-6255`）。結果是累加器從未歸零、輸出是垃圾、零診斷。

→ 所以展開要**保留迴圈外殼**，把多份 body 串在迴圈裡（迴圈實際只跑一次）。現成模板是 `KernelWriter.py:5851-5873` 的 `needSecondLoop`（`skipClose=True` + 手寫的 copy 間 dec/cmp/branch）。

保留外殼還有一個好處：`RegionClonePass` 的區域邊界是「最後一個第一次出現的 src-C 累加器」，而 C 累加器**跨 k-tile 共用**，所以 `seenAccs` 在第 0 份的前半就飽和，clone 區域自然只涵蓋第 0 份。上次組語量到的「16 個歸零 WMMA」正好等於 MIWaveTile 2×8 的 16 個 MI tile，這個推論有實測背書。

**D. 標籤撞名不是展開的阻礙。** unroll loop body 幾乎完全沒有固定名稱的標籤（`globalReadDo` / `localWriteDo` / `localReadDo` / `mfmaIter` 都不產生標籤），只有 `closeLoop` 有四個，而本 config 只會發出 `LoopEndL`、還只在最後一份發。（peel 造成的撞名是另一回事，見步驟 4b。）

**E. StinkyTofu 的區域是靠 rocisa module 名稱 `"loopBody"` 界定的**，不是靠標籤或 CFG 形狀（`conversion/rocisa/ToStinkyTofuUtils.cpp`）。`postMainLoopBarrierCheckAndReset` 在 `PGR>=2` 時本來就把回邊建模關掉、退化成線性走訪（`KernelWriter.py:11081`）。所以多份 body 對這兩個後端機制是安全的。

---

## 3. 貫穿全程的驗收閘門

因為模擬器看不見同步性錯誤，**正式驗收 = 驗證 PASSED 且 組語結構比對通過**。兩者缺一不可。

### 3.1 組語結構比對（主要閘門，必做）

寫一支可重跑的腳本（建議 `tools/rap_asm_check.py`），從產出的 `.s` 切出 iter0 與 iterN 兩段（邊界：`label_PersistentLoopStart` → `label_RAP_IterN` → join 標籤），對每段統計並比對：

| 指標 | 步驟 4b 之後（peel，尚未丟指令） | 步驟 5 之後（已丟 A/MXSA） |
|---|---|---|
| `s_wait_dscnt` 數量 | 相等，且皆 > 0 | 相等，且皆 > 0 |
| `s_wait_tensorcnt` 數量 | 相等 | 相等 |
| `s_barrier` 數量 | 相等 | 相等 |
| `v_wmma*` 總數 | 相等 | 相等 |
| **src C = 0 的 WMMA 數量** | 相等，且 > 0 | 相等，且 > 0 |
| `tensor_load_to_lds` 數量 | 相等 | 相等（指令仍發出，只是 descriptor 被清零） |
| ValuA / ValuMXSA 的 `ds_load` | 相等 | iter0 > 0，**iterN == 0** |
| ValuB / ValuMXSB 的 `ds_load` | 相等 | 相等 |

這張表的每一列都對應到上次一個真實的 bug：`s_wait_dscnt` 那列抓 CFG 前驅缺失、`s_barrier` 那列抓 barrier token 起始狀態、`src C = 0` 那列抓 `CloneSpec` 名稱耦合。

另外掃描**整份 kernel**有沒有任何指令寫入常駐 VGPR 區間。掃描器必須同時比對 `v[N]` 括號形式與**裸 `vN` 形式**——上次就是因為只比對括號形式，讓一個錯誤結論撐了好幾輪（store 的位址計算用的是 `v_add_nc_u32 v2` 這種裸寫法，而 MXSA 常駐在 v0–v11）。

### 3.2 codegen 時期檢查（輔助，必做）

產完每個區段之後走一遍該區段的 module，用 `isinstance(inst, DSLoadInstruction)` 數出實際發出的 A/MXSA local read 數量，跟 `numReadsPerIter*` 的假設比對，不一致就讓 codegen 失敗。走訪方式抄 `postMainLoopBarrierCheckAndReset`（`KernelWriter.py:10933`）。

這道檢查上次救了自己兩次：一次擋下「只丟一個 operand 卻把兩個計數都歸零」的不一致組合，一次在把 MXSA 改成保留時立刻報錯。它抓不到後端造成的差異（那是 3.1 的職責），但它抓得到 Python 端的不一致，而且是在**建置當下**爆。

### 3.3 回歸

每個檢查點跑一次完整 unit test。上次的基準是 6223 passed、757 snapshots passed；`test_gl2_prefetch_offset.py` 的失敗是缺 `hip` python binding 的既有環境問題，與本改動無關。

### 3.4 對照組保存

因為決定「全部做完再一次 commit」，每個綠燈檢查點要手動保存兩樣東西到 **repo 外**：

- 產出的 `.s`（下一步逐字 diff 的對照組——上次定位六個 bug 靠的就是這個）
- `git diff` 落成 patch 檔

建議放 `/home/josh/rap_refs/stepN/`。（若之後想要保護又要乾淨的歷史，「每步 WIP commit、最後 squash 成一個」可以兩者兼得。）

---

## 4. 里程碑 M1：重建到上次的最終狀態

M1 結束時 `k` 仍然等於 `PGR + 1 = 3`、`K = 768`，跟上次收工時相同。

### 步驟 1：參數接線

抄 `PrefetchAcrossPersistent` 的既有慣例，四個地方：

1. `Tensile/Common/ValidParameters.py` — 加 `"ReuseAcrossPersistent": [0, 1]`
2. `Tensile/Common/GlobalParameters.py` 的 `defaultBenchmarkCommonParameters` — 加 `{"ReuseAcrossPersistent": [0]}`
3. `Tensile/Common/RequiredParameters.py` 的 `getRequiredParametersMin()` — **必加**。否則 RAP=0 與 RAP=1 會 hash 出同一個 kernel 名字，其中一個被當重複靜默丟掉，`[0,1]` fork 會變成兩筆同一個 kernel 的數據
4. `Tensile/SolutionStructs/Solution.py:1863` 的「StreamK 關閉時歸零」區塊 — 加 `state["ReuseAcrossPersistent"] = 0`，避免產生兩個完全一樣的 kernel

命名採**無條件**加入（`RAP0` / `RAP1` 都出現在 kernel 名稱裡，不是只有啟用時才帶）。這是上次明確的要求，代價是**所有** kernel 名稱都會變、`.ambr` snapshot 要重生（上次共 52 個檔）。

`SizeMapping.StateKeys`（`Contractions.py`）**不用改**，RAP 純粹是 codegen 時期的開關。

比照 `Tensile/Tests/unit/test_PrefetchAcrossPersistent.py` 加一個註冊測試。

**驗收**：yaml 跑得完，產出四個名字不同的 kernel（兩個 config × RAP0/RAP1），`Number of duplicate kernels: 0`，全部 PASSED。此時 RAP=1 與 RAP=0 的組語應該**完全相同**。

### 步驟 2：predicate 與 reject 條件

**2a. reject 條件區塊**，放在 `Solution.py:1776` 的 PAP guard 旁邊，每條要有獨立訊息：

- `PrefetchAcrossPersistent == 1`
- `StreamK == 3` 且 `StreamKForceDPOnly == 1`
- `InnerUnroll == 1`
- `NoTailLoop == True`（訊息裡寫明是 `ASEM % DepthU == 0`；注意單一個 `%` 不要寫成 `%%`）
- `DirectToVgprA == False`、`ExpandPointerSwap == False`
- `enableTDMA and enableTDMB and NumWaves > 1`（wave 分工的前提，見 `KernelWriterAssembly.py:354` 的 `isTdmWaveSeparated`）
- 常駐暫存器算出來放不下 → reject，**絕不 silently 退回 RAP=0**

**注意：`ceil(M/MT0) == 1` 和 `batchCount == 1` 不可能是 reject 條件**，因為 solution 推導時不知道 problem size。它們必須走 runtime predicate。這是上次計畫文件裡的一個錯誤，這裡直接修正。

**2b. runtime predicate**，在 `Contractions.ProblemPredicate.CompoundPredicates`（`Contractions.py:508`）的 RAP 區塊發三條：

```yaml
- {index: 0, type: SizeEqual,    value: MacroTile0}   # M == MT0
- {index: 1, type: SizeMultiple, value: MacroTile1}   # N % MT1 == 0
- {index: 3, type: SizeEqual,    value: k * DepthU}   # K
```

- M 用 `SizeEqual` 而不是 `SizeLessThan(MT0+1)`：後者只擋「M 不超過一個 tile」，但 M=32 這種情況那唯一的 tile 只填一半、照樣走 masked store。相等一條就同時保證單一 M-tile 與 M 方向無邊界
- N 的 `SizeMultiple` 保證 edge=0，讓 v0 收斂在 `edge=0 + beta=1` 這一檔
- `SizeEqual` / `SizeMultiple` 在 C++ runtime 端**已註冊且已實作**（`include/Tensile/ContractionProblemPredicates.hpp`、`Serialization/ContractionPredicates.hpp`），零改動
- **不要**新增 `AssertSizeEqual` 這類使用者參數。K 從既有參數推導，調參端不必寫任何東西
- 索引慣例：0=M、1=N、2=batch、3=K

不符時 standalone client 會報 `DID_NOT_SATISFY_ASSERTS` 並跳過該 kernel。

**2c.** 在 `KernelWriter.py` 加 `isReuseAcrossPersistentEnabled(kernel)`，比照 `:10831` 的 `isPrefetchAcrossPersistentEnabled` 自己重算 enable 條件，不信任 state flag。

**驗收**：正向 `(64, 32768, 4, 768)` 被選中；負向三筆——K 不合、`M > MT0`、`N` 不整除 `MT1`——RAP=1 都回報 `DID_NOT_SATISFY_ASSERTS` 而 RAP=0 照常執行。

### 步驟 3：常駐 VGPR 配置

**核心是一行**：`KernelWriter.py:7197` 的 `numVgprBuffer`。RAP 開啟時，**只有 A 與 MXSA** 的 buffer 數從 `LoopIters`（2）變成 `LoopIters × k`。既有的 sizing 公式與 `vgprValuA_X{n}_I0` 的命名會自動跟著長大。

k=3 的具體數字：ValuA 64 → 192（6 組 × 32），ValuMXSA 4 → 12（6 組 × 2）。

**五個借用站點必須全部處理，這是整個功能最容易漏的地方（上次漏了兩次）**：

| # | 位置 | 動作 | RAP 要做的事 |
|---|---|---|---|
| 1 | `KernelWriterAssembly.py:2870` | `setupNewTile`：`vgprPool.add(0, lastValuMXSAB, "ValuMXSAB")` | 排除常駐 MXSA 區間 |
| 2 | `KernelWriterAssembly.py:2876` | `setupNewTile`：`vgprPool.add(a.startVgprValu, …, "ValuAB")` | 排除常駐 ValuA 區間 |
| 3 | `KernelWriterAssembly.py:6257` / `:6266` | `initC`：對應的兩個 `remove` | 跟著排除，**add/remove 必須對稱** |
| 4 | `KernelWriter.py:6091` | 主迴圈後回收 ValuAB | 只還 ValuB，起點改用 `b.startVgprValu` |
| 5 | `KernelWriter.py:6643` | 主迴圈後回收 ValuMXSAB | 起點改成常駐區塊的尾端 |

站點 1、2 是 `setupNewTile` 借 Valu 區當 scratch（註解寫「C regs are not used during initialization」），而 `setupNewTile` **每個 persistent iteration 都會執行**，所以不排除就會毀掉常駐資料。

站點 5 是上次最後才找到的元凶：它把整個 MXS 區塊 `[0, lastValuMXSAB)` 還給 store，而常駐 MXSA 住在 v0–v11，正是 store 位址計算最愛用的低位暫存器（`v_add_nc_u32 v2`、`v_add_co_u32 v1` 各出現 252 次）。

若站點 3 的 `remove` 沒跟著改，會出現 `RegisterPool::remove … already unavailable` 的 warning。**那個 warning 是有意義的訊號，不要忽略它。**

**對齊**：常駐 A 區塊要對齊到**一個 A 運算元的寬度**（本 config 是 16），因為 `s_set_vgpr_msb` 的高位是從區塊起點算的，單一運算元跨過 256 的倍數會定址錯誤。不要用寫死的 32——上次實測 32 對齊在 MT64x512 上浪費 38 個暫存器，把 store 餘裕從 134 壓到 130（`E = 128`）。`KernelWriterAssembly.py:6757` 有個 `valuVgprAlignment = 8 if HasVgprMSB else 2` **算了但整棵樹沒人用**，底下還是傳字面的 2 — **不要動它**，接上它會改變所有 gfx1250 kernel 的配置。另開 issue。

**MXSA 必須留在 v0–v255**：這已經是現有 layout（`vgprMXSBase = 0`），RAP 只是把 block 從 4 撐大到 12，遠在 256 以內。

**驗收**：四個 kernel 都 PASSED（此時只是多佔暫存器，行為應與 RAP=0 完全相同）。組語裡 ValuA 有 `X0`–`X5` 六組、MXSA 同樣六組、`vgprBase` 對齊 16。`RegisterPool` 零 warning。記錄組語註解裡 `elementsPerBatch=` 的實際值。

### 步驟 4a：k-tile 索引

buffer 索引改用**絕對迭代索引**：MFMA 用 `rapKTileIdx * LoopIters + u`，local read 用 `rapKTileIdx * LoopIters + u + pflr`（讀取跑在前面一格）。

`rapKTileIdx` 是**產生期常數**，按區段指定：unroll loop body = 0、NGLL = `k−2`、NLL = `k−1`。依上次的要求，每個區段要加註解標明它屬於哪一類，例如 `/* RAP k-tile 0/3 (loop body, issues global loads) */`。

**一個會覆蓋常駐資料的陷阱**：local read 跑在 MFMA 前面一格，所以最後一個 k-tile 的最後一次讀取，絕對索引會超出配置範圍並繞回 X0——也就是常駐的 k-tile 0。那次讀取的語意是「預取下一個 tile 的第一個 k-tile」，正是 RAP 要消除的搬移，所以**抑制它是正確的**。上次實測既有的 `doReadA` 條件本來就不會發出那次讀取，但這道防護要保留，因為它是唯一擋住那個覆蓋的東西，而且一旦 PLR 或 LoopIters 改變就會變成必要的。

**`rapKTileIdx` 必須進入步驟 4b 的狀態快照範圍。** 上次它掛在 writer 上、不在 `self.states` 裡，導致 iterN 沿用 iter0 結束時的值（2），絕對索引超出範圍後整批讀取被抑制。這是上次的第一個 bug。

**驗收**：四個 kernel PASSED；組語顯示 WMMA 讀遍 `X0`–`X5`，總指令數與基準完全一致——**指令一個都沒變，只是換了目標暫存器**。

### 步驟 4b-1：純重構

把 `kernelBody` 裡從 `setupNewTile` 到 NLL 結束的約 590 行抽成 `_persistentComputeSection`。邊界很乾淨：只有 `expand`、`module`、`tPM` 進去，只有 `pack` / `packPre` 出來，而且區塊本來就在方法本體的縮排層級，所以是純粹的區塊搬移。

**驗收**：把重構前的組語存起來逐字 diff，四個 kernel 的**指令流必須完全相同**。這一步不允許任何指令變化。

### 步驟 4b-2：iter0 / iterN 剝離

把 `_persistentComputeSection` 呼叫兩次，中間插入分支與 join 標籤：

```
label_PersistentLoopStart:        ← 只有 iter0 從這裡進入
  iter0 計算段
  s_cbranch (永遠成立的條件分支) → label_RAP_StoreJoin
label_RAP_IterN:                  ← 後續 iteration 的迴圈頭
  iterN 計算段
label_RAP_StoreJoin:
  store（共用，佔全檔約 84%，絕對不能複製）
  close persistent loop → 回跳 label_RAP_IterN
```

**這一步上次踩了六個坑，全部列在下面，逐一預先處理：**

1. **`rapKTileIdx` 未還原** — 見步驟 4a
2. **標籤撞名** — 計算段裡有一批**固定名稱**的標籤（`SkipStaggerA`、`skipPGR2_1`、`openLoopL`、`LoopBeginL` 等）。`self.labels.getName()` 那一族是**刻意**產生確定性名稱好讓分支和目標對得上，發射兩次必然衝突，CFG builder 會直接 abort。加一個 `rapLabelSuffix` 機制（非 RAP 時後綴為空字串，其他 kernel 的輸出逐字不變），約需包 8 個 `getName` 站點與 21 個固定字串 `Label(...)` 站點
3. **無條件分支讓 iterN 在 CFG 裡沒有前驅** — CFG builder 的規則是「無條件分支不產生 fall-through 邊，條件分支才會」。用 `s_branch` 跳過 iterN 會讓它被排除在後端資料流分析外，**一個 waitcnt 都不會被插入**。改成永遠成立的**條件分支**
4. **barrier 重建的起始 token 狀態** — `postMainLoopBarrierCheckAndReset` 線性走訪 token 狀態重建 barrier，但 iterN 是 back-edge 目標而不是 iter0 的接續，會繼承 iter0 的結束狀態。要在 `label_RAP_IterN` 把 token 狀態還原成 persistent loop 入口的狀態
5. **狀態快照必須是 deep copy** — `freeSgprVarPool`、`lraTileProperties` 這些**可變容器**若只存參照，iter0 就地改掉之後「還原」等於還原一個已經被改壞的物件，iterN 會配到不同的暫存 SGPR。大型唯讀表格仍可共用參照。`saveLocalPointers` / `restoreLocalPointers`（`KernelWriterAssembly.py:13404`）是既有的模板
6. **`CloneSpec` 的轉換靠名稱識別** — iterN 需要自己的 clone job（`startLabel` 指向加了後綴的標籤），但 `name` **必須維持 `"InitCIterWmma"`**。上次把它命名成 `InitCIterWmmaRAPIterN`，區域確實被複製了、`label_InitCIterWmmaRAPIterN_..._1` 也出現在組語裡，**但把 WMMA 的 src C 改寫成 0 的轉換沒有套用**，於是 C 從第二個 tile 起就不再歸零

另外：`SkPrefetchPrimed` 的分支要保留在兩段裡——它會在某些 slice 跳過 NLL 時被防禦性清 0，所以「iterN 進來時一定是 1」並不成立，不能靜態化掉。

**若卡住，用分階段診斷。** 上次解開這題的關鍵是一個中間態：**讓兩份副本純粹接續（fall-through），不加分支也不加 iterN 標籤**，其他一切不變。這一版計算結果一定是錯的（兩份各自推進 tile 索引、只 store 一次），但它能回答唯一重要的問題——後端會不會對第二份副本插 waitcnt。上次這一步一次就把問題砍半：純接續時兩份都拿到了完整的 waitcnt，立刻排除「複製本身有問題」，把範圍鎖死在分支與標籤結構上。**建議一開始就把這個模式做成一個開關保留下來。**

**驗收**：四個 kernel PASSED，**且 §3.1 的組語結構比對用「步驟 4b 之後」那一欄全部相等**。此時 iterN 仍然照常載入 A，只是程式碼被複製了一份。

### 步驟 5：iterN 丟掉 A/MXSA 的搬移

**5a. 刪掉 A/MXSA 的 `ds_load`**（只在 iterN 那次發射）。

**5b. 讓偶數 wave 的 A/MXSA TDM 失效。** 抄 `KernelWriterAssembly.py:11402-11407` 的 HalfPLR 前例：

```
s_bitcmp1_b32 s[sgprWaveIdx], 0     // check wave parity
s_cmov_b32 s[sgprtdmAGroup0+0], 0   // even wave: NULL A descriptor
s_cmov_b32 s[sgprtdmMXSAGroup0+0], 0
```

依上次的要求，比較指令與 cmov 要相鄰出現以利閱讀。四個要點：

- **不能只設一次**。descriptor 每個 persistent iteration 都會被重建（組語裡 `s_mov_b32 s[sgprtdmAGroup0+0], 1` 出現在 `PersistentLoopStart` 之後，還有一處標著 `restore PAP LDS bank after descriptor refresh`）。cmov 必須跟在**每一次 descriptor 重建之後**
- B/MXSB 的 descriptor 是**別名**到 A 的 SGPR（`RegSet("s", "sgprtdmBGroup0", "sgprtdmAGroup0")`），奇數 wave 拿到的是 B 的內容，所以只對偶數 wave 清零，B 完全不受影響
- SGPR 名字是 **`sgprWaveIdx`**，而且它會被 UNDEF 回收。**必須透過 `_emitTdmWaveParitySCC`（`KernelWriterAssembly.py:19167`）取得 parity**，不要直接讀 `sgpr("WaveIdx")`
- TDM 的 wait 是獨立的 `s_wait_tensorcnt`，非 subtile 路徑上每個 call site 傳的都是 0（full drain），所以清零 descriptor 不影響 wait 正確性

**5c. `numReadsPerIter*` 依區段給不同的值**（靠 4b 的 snapshot/restore 機制才做得到 per-region），並加上 §3.2 的 codegen 時期檢查。

**驗收**：四個 kernel PASSED，**且組語結構比對用「步驟 5 之後」那一欄通過**。組語裡 iterN 的 A/MXSA `ds_load` 為 0、`ds_load` 總數應下降約 21%（A+MXSA 在 ds_load 裡的占比）。

### 步驟 6：store 中立性守門與收尾

**6a. store 中立性守門。** 在 `KernelWriterAssembly.refineOccupancy`（`:16193`）算完 `numElementsPerBatch` 之後插入：用 `numVgprAvailable + W` 還原出「沒有常駐時的可用量」再算一次批次數，變多就設 `overflowedResources`（新錯誤碼），kernel 換成 `s_endpgm` stub 並在 `PrintSolutionRejectionReason` 開啟時印出理由。

```
W = (b.startVgprValu − a.startVgprValu) + (mxsa.startVgprValu + mxsa.numVgprValu)
```

`W` 抽成 `KernelWriter.rapStoreWithheldVgprs`，讓步驟 3 的尾端回收與守門共用同一個定義，避免兩邊漂移。只擋 `beta and not edge` 那一個變體——beta=1 每個元素多讀 C，比 beta=0 緊；N 的 predicate 已經保證 edge=0。

reject 訊息要把上限一起算出來，例如「holds N vgpr resident, splitting the store into 2 batches instead of 1; largest store-neutral K is X but this kernel needs Y」。

**一個已知限制要寫進註解**：`numVgprAvailable + W` 還原的是「同一顆 kernel 但把常駐區塊還回去」，要等同真正的 RAP0 baseline，前提是 occupancy 沒變（`maxVgprs` 來自 `setOccupancy`）。本 config 的 occupancy 已被 LDS 釘在地板，所以不跨界。

**6b. 收尾**：重生 `.ambr` snapshot、移除所有開發期的 debug 環境變數與臨時鷹架、清掉引用對話編號的註解、把 `mxf8mxf4_gfx1250_rap/` 輸出目錄加進 `.gitignore`。

**驗收**：完整 unit test 綠燈；守門的 reject 分支用 unit test 直接呼叫驗證（中立情況通過、退化情況拒絕並檢查訊息裡的兩個 K 值、beta=0 / edge=1 不會被誤擋）。

**M1 完成 = 上次收工時的狀態。**

---

## 5. 里程碑 M2：解開 K 的上限

上次 K 被鎖在 768，是**三條**限制相乘的結果，展開 unroll loop 只解開其中兩條：

| 想走的路 | 擋住它的東西 | M2 之後 |
|---|---|---|
| PGR=3（k=4） | `Stream-K + TDMInst=3 requires PrefetchGlobalRead in (1, 2)` | 不再相關，PGR 只管預取深度 |
| 同上 | `PrefetchAcrossPersistent requires PrefetchGlobalRead in [1, 2]` | 不再相關 |
| DepthU=512 + PGR=1（k=2） | `ReuseAcrossPersistent requires NoTailLoop`（ASEM 合法上限 256，而 `ASEM % DepthU == 0`） | **仍然存在**，DepthU 壓在 256 |

所以 M2 之後 K 的粒度仍是 256 的倍數，但 k 脫離 PGR，上限改由 store 中立性決定。

### 步驟 7：`k_max` 解析模型

在 `Solution.assignDerivedParameters` 算出 `k_max` 並存進 state（例如 `_RAPNumResidentKTiles`），predicate 由它發（`SizeEqual(index=NumIndicesC) = k_max × DepthU`）。

```
k_max = floor( (avail0 − E × V) / R )
  E = ValuC 元素數 / StoreVectorWidth        （MT64x512 → 128，MT64x256 → 64）
  V = numVgprsPerElement（beta=1, edge=0 那一檔）
  R = 每 k-tile 的常駐量 = (ValuA per block + MXSA per block) × LoopIters   （本 config = 68）
  avail0 = 推導階段對「沒有常駐時 store 可用暫存器」的估計
```

`avail0` 是唯一需要建模的量（推導階段拿不到 pool 的真實狀態），所以這個模型**必然是估計值**。安全性由步驟 6a 的守門保證：估太大 → 守門 reject 並印出真正的上限；估太小 → 只是保守，不會算錯。

再加一個 **debug-only 的 `k` 覆寫開關**（環境變數即可），用來把 k 從 1 掃到 `k_max + 1`，讓守門的 reject 分支能被**真實 kernel** 走到。上次那條分支在真實 config 下走不到，只能靠 unit test 驗，這是缺口。

**驗收**：兩個 config 自然產出不同的 `k_max`（預期 MT64x512 約 3、MT64x256 約 5）；覆寫到 `k_max + 1` 時守門正確 reject 並印出上限；覆寫到 `k_max` 以下時仍然 PASSED。

### 步驟 8：(k − PGR) 份 loop body

**保留迴圈外殼**，把 `(k − PGR)` 份 body 串在迴圈裡。迴圈實際只跑一次（counter 從 `ItersPerTile = k` 每份減 1，跑完 `k − PGR` 份之後等於 `PGR` 而退出），所以 `label_LoopBeginL`、回邊、`InitCIterWmma` 的錨點、`RemoveDscntPass` 的標籤判斷全部原封不動。

現成模板：`KernelWriter.py:5851-5873`（`needSecondLoop` / `UnrollLoopSwapGlobalReadOrder`），用 `skipClose=True` 加手寫的 copy 間 dec/cmp/branch。

`rapKTileIdx` 的區段對應變成：第 j 份 body = `j`（j 從 0 到 `k−PGR−1`）、NGLL = `k−2`、NLL = `k−1`。

**每份 body 之間必須推進的狀態**（上次 peel 的教訓在這裡會再出現一次，只是規模更大）：

- LDS memory token：`ldsTensorTokenIdx`、`ldsReadTokenIdx`、`ldsWriteTokenIdx`
- LDS 雙緩衝位址：`localReadSwapByteOffset` / `localWriteSwapByteOffset`（`ExpandPointerSwap=False` 時是發 `v_xor`，不是編譯期偏移，所以 emitter 不用追 parity）
- `tP["localReadOffset"]` 與 `states.localReadDoCnt*`
- `states.perIterLocalWriteCanSkip`、`self.codes.*`（每次 `makeSchedule` 重建）

`saveLocalPointers` / `restoreLocalPointers`（`KernelWriterAssembly.py:13404`）與 `KernelWriter.py:5881-5883` 的 copy 間 token swap 是既有模板。

**一個要驗證的假設**：NGLL / NLL 是**發射一次**的，它們假設進入時 LDS 雙緩衝處於某個特定 phase，而 phase 取決於 `(k − PGR) mod 2`。k=3（1 份）與 k=4（2 份）的 parity 相反。Python 端的 emitter 是循序推進 token 的，所以理論上會自動跟上，但這正是上次咬人的那類問題——**k 從 3 加到 4 的那一步要單獨當一個檢查點**，不要一次跳到 k=5。

**驗收**：k 從 3 逐步加到各 config 的 `k_max`，每一階都要 PASSED 且組語結構比對通過；yaml 裡放多個 K（768 / 1024 / 1280 / 1536），讓 predicate 自己挑，不符的回報 `DID_NOT_SATISFY_ASSERTS`。

---

## 6. 已排除的假設（不要重查）

上次在 MXSA 常駐那題上繞了很久，這些假設都已經被證據排除：

| 假設 | 排除方式 |
|---|---|
| TileSpan 讓 MXSA 的常駐語意跟 A 不同 | **錯**。真正原因是 `KernelWriter.py:6643` 把 `[0, lastValuMXSAB)` 還給 store，而 MXSA 住在 v0–v11 |
| iter0 沒有把 X0–X5 全部填滿 | 排除，組語確認六組都填了 |
| 兩次 `ds_load` 寫同一個 X1 互相覆蓋 | 排除，那是 `InitCIterWmma` clone 的**替代路徑**，執行期只走其中一條，不是循序執行 |
| `s_wait_dscnt` 低估 | 排除，兩段數量相同且 iterN 的值更保守 |
| `numReadsPerIter` 歸零造成排程問題 | 排除，保留不歸零仍然失敗 |
| barrier 缺失 | 排除，兩段各 12 個 |
| iter0 與 iterN 的 WMMA 指令不同 | 排除，差異只是重排，且兩個 build 的差異完全相同而其中一個 PASS |
| `matrix_a_scale:1` 對 `ds_load` 有額外依賴 | 排除。規格：`SCL_OPSEL[0]`（bit 11）只是選 scale VGPR 的哪一半 lane（`:0` 用 lane 0–15，`:1` 用 lane 16–31），跟資料來源無關 |

**方法論教訓**：掃描「誰寫入常駐區間」時，必須同時比對 `v[N]` 括號形式與**裸 `vN` 形式**。上次只比對括號形式，讓一個錯誤結論撐了好幾輪。

---

## 7. 風險清單

| 風險 | 徵兆 | 對策 |
|---|---|---|
| **模擬器看不見同步性錯誤** | 驗證 PASSED 但 kernel 在真實硬體上會錯 | §3.1 的組語結構比對，**強制** |
| 常駐暫存器被借去當 scratch | 結果錯誤但指令看起來完全正確 | 步驟 3 的五個借用站點全部處理；`RegisterPool` warning 當作訊號 |
| `InitCIterWmma` 沒套用到某一份副本 | C 不歸零，結果錯 | 組語結構比對的「src C = 0 的 WMMA 數量」那一列 |
| iterN 在 CFG 裡沒有前驅 | 完全沒有 waitcnt，模擬器上仍 PASSED | 用條件分支；組語結構比對的 `s_wait_dscnt` 那一列 |
| `numReadsPerIter*` 與實際指令數不一致 | 真實硬體上偶發 `-nan` | §3.2 的 codegen 時期檢查，**強制** |
| store 跳成兩批 | 組語的 `elementsPerBatch=` 掉到 `E` 以下 | 步驟 6a 的守門 |
| 常駐區塊跨 256 邊界 | 隨機錯誤結果 | 對齊到一個 A 運算元的寬度 |
| cmov 只設一次被 descriptor 重建蓋掉 | A 仍在搬、結果正確但沒有效益 | 檢查組語中 cmov 的出現次數與位置 |
| **WG 跨 batch 導致常駐 A 過期** | 步驟 5 在 batch=4 上 FAILED | 見下 |
| 成果再次遺失 | — | §3.4 的 repo 外備份；考慮改成「每步 WIP commit、最後 squash」 |

### batch 的應變方案

上次留下一個未解的矛盾：靜態證據顯示 iterN 完全不更新常駐 A、且 A 逐 batch 不同、且測試確實跨了 batch 邊界，但全量驗證仍然 PASSED。上次的裁決是「先當作 v0 已經支援任意 batch」。

**這次的測試 problem（batch=4，且 WG 走的 tile 是 `flat += skGrid`）本身就是那個決定性實驗。** 不需要另外設計，只要步驟 5 通過就是證據，失敗就是答案。

若步驟 5 在 batch=4 上 FAILED：

1. 先用 batch=1 的 size 確認失敗確實來自 batch（而不是步驟 5 本身的 bug）
2. 若確認是 batch，短期加 `BatchSizeEqual = 1` 的 guard（`Contractions.py:511` 已有 emission 路徑，只要在 RAP 的推導裡寫入這個 state key），把任意 batch 留到 v1
3. v1 的兩條路：讓 tile 映射保證 WG 不跨 batch（改 StreamK 分配，影響面大），或偵測到 batch 改變時重新載入 A（該次 iteration 退化成非 RAP，需要 runtime 分支）

---

## 8. 明確排除在 v0 之外

- `ceil(M/MT0) > 1` — 組語證據：`s_mul_i32 s86, s[sgprStrideA0I], 64` / `s_mul_i32 s86, s86, s[sgprWorkGroup0]`，A 的位址跟 `WorkGroup0` 成正比，`nWG0 > 1` 時同一個 WG 相鄰兩個 tile 的 A 就換了一份
- K-split（StreamK 非 DP-only）
- `DepthU > 256`（被 ASEM 上限擋住，非本計畫可解）
- 修 ASEM 被降成 32 的既有問題 — **獨立 issue**
- 接上 `valuVgprAlignment` 死變數 — **獨立 issue**
- subtile 路徑（`UseSubtileImpl=1`）— `Components/Subtile/InstructionEmitter.py` 的 tensorcnt 是數實際發出的指令，需要另外處理
- A 繞過 LDS 直接 global→VGPR（DirectToVgpr 風格）
- 真正的直線展開（拿掉 `openLoop`/`closeLoop`、改用 `acc2_imm=0` 歸零 C）— 架構上更乾淨，會移除 `InitCIterWmma` 這個脆弱耦合，但動到 C 歸零路徑，風險等級不同。`rocisa` 已支援（`instruction/mfma.hpp` 的 `acc2_imm` overload）、Subtile 已在用（`Components/Subtile/Kernel.py:920-929`）、`mfmaIter`（`KernelWriterAssembly.py:8903`）那個從未被讀取的 `firstIter` 參數就是留給它的位置。**獨立 issue**

---

## 9. 程式碼位置速查表

行號基於 `b1df07af887`，全部已在目前的 checkout 驗證存在。

### 參數與 predicate

| 用途 | 位置 |
|---|---|
| `validParameters` | `Tensile/Common/ValidParameters.py` |
| `defaultBenchmarkCommonParameters` | `Tensile/Common/GlobalParameters.py` |
| `getRequiredParametersMin()` | `Tensile/Common/RequiredParameters.py` |
| kernel 命名 | `Tensile/SolutionStructs/Naming.py` |
| PAP guard block（reject 模板） | `Tensile/SolutionStructs/Solution.py:1776` 起 |
| StreamK 關閉時歸零子參數 | `Tensile/SolutionStructs/Solution.py:1863` |
| `NoTailLoop` 推導 | `Tensile/SolutionStructs/Solution.py:4513-4517` |
| `reject()` | `Tensile/SolutionStructs/Utilities.py` |
| `CompoundPredicates` | `Tensile/Contractions.py:508` |
| `problemPredicate` 建構 | `Tensile/Contractions.py:876` |
| runtime predicate 實作 | `include/Tensile/ContractionProblemPredicates.hpp` |
| client 執行 predicate | `client/src/SolutionIterator.cpp` |

### VGPR 配置與生命週期

| 用途 | 位置 |
|---|---|
| `numVgprBuffer` | `Tensile/KernelWriter.py:7197` |
| `setupNewTile` 借 MXSAB 當 scratch | `Tensile/KernelWriterAssembly.py:2870` |
| `setupNewTile` 借 ValuAB 當 scratch | `Tensile/KernelWriterAssembly.py:2876` |
| `initC` remove MXSAB / ValuAB | `Tensile/KernelWriterAssembly.py:6257` / `:6266` |
| 主迴圈後回收 ValuAB | `Tensile/KernelWriter.py:6091` |
| 主迴圈後回收 ValuMXSAB（**上次的元凶**） | `Tensile/KernelWriter.py:6643` |
| `valuVgprAlignment` 死變數 | `Tensile/KernelWriterAssembly.py:6757` |
| `refineOccupancy`（store 批次） | `Tensile/KernelWriterAssembly.py:16193` |

### 迴圈結構與後端

| 用途 | 位置 |
|---|---|
| `_loopBody` | `Tensile/KernelWriter.py:4200` |
| 多份 body 的現成模板（`needSecondLoop`） | `Tensile/KernelWriter.py:5851-5873` |
| `openLoop` / `closeLoop` | `Tensile/KernelWriterAssembly.py:7790` / `:7922` |
| `InitCIterWmma` 的 `CloneSpec` | `Tensile/KernelWriter.py:6836-6839` |
| `RegionClonePass` 實作 | `shared/stinkytofu/src/transforms/asm/RegionClonePass.cpp` |
| `initC` 跳過 `v_mov` 歸零的條件 | `Tensile/KernelWriterAssembly.py:6243-6255` |
| `postMainLoopBarrierCheckAndReset` | `Tensile/KernelWriter.py:10933`（PGR≥2 的線性化在 `:11081`） |
| `saveLocalPointers` / `restoreLocalPointers` | `Tensile/KernelWriterAssembly.py:13404` |
| `mfmaIter`（含未使用的 `firstIter`） | `Tensile/KernelWriterAssembly.py:8903` |

### TDM 與 wave 分工

| 用途 | 位置 |
|---|---|
| `isTdmWaveSeparated` | `Tensile/KernelWriterAssembly.py:354` |
| HalfPLR 清零 descriptor 的前例 | `Tensile/KernelWriterAssembly.py:11402-11407` |
| `_emitTdmWaveParitySCC`（取得 wave parity） | `Tensile/KernelWriterAssembly.py:19167` |
| TDM 指令發射 | `Tensile/Components/TensorDataMover.py` |
| `isPrefetchAcrossPersistentEnabled`（predicate 模板） | `Tensile/KernelWriter.py:10831` |

### 跨階段管線（M2 用）

| 用途 | 位置 |
|---|---|
| `passPostKernelInfoToSolution`（codegen → state 回寫） | `Tensile/TensileCreateLibrary/Run.py:298-310` |
| `TensileCreateLibrary` 的 predicate 早於 codegen | `Tensile/TensileCreateLibrary/Run.py:1087` vs `:1101` |
| joblib worker 行程邊界 | `Tensile/Common/Parallel.py:220`（單執行緒）/ `:239`（平行） |
