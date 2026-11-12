# 10 篇 ICLR 2026 相关论文的叙事结构与语言风格分析

检索与核实日期：2026-09-08。分析对象：NQRL 当前稿件的写作方式。

## 1. 主要发现

这组论文支持 NQRL 采用直接、主动的作者口吻：肯定已有研究的价值，明确指出一个具体问题，解释自己的设计如何回应这个问题，再用实验呈现收益。机制解释通常是论证的一部分，不需要每讲一遍理由，就补一句“这没有被证明”或“这不保证有效”。

但“主动表达”也不等于把不同性质的判断写成同一种结论。它们通常通过句子的动词和上下文区分：我们设计了什么、实验观察到了什么、我们如何解释观察，以及在什么假设下可以证明什么。读者因此能理解论据强度，而不需要作者反复自我否定。

以下是有针对性的样本分析，不是对全部 ICLR 2026 论文风格的统计调查。章节主线是按论证功能压缩的归纳，不是逐字目录。

## 2. 论文选择与阅读来源

10 篇均在 ICLR 2026 官方 proceedings 的 Conference 条目中核实，未以 workshop 或仅投稿论文凑数。选择覆盖拟度量 GCRL、目标表示、在线探索、分离表示，以及价值梯度引导优化。

| 编号 | 论文 | 与 NQRL 的关系 | 本次阅读全文的来源 |
| --- | --- | --- | --- |
| P1 | [Scaling Goal-conditioned Reinforcement Learning with Multistep Quasimetric Distances][p1] | 同属拟度量 GCRL；直接参考如何写已有方法的改进 | 官方 PDF |
| P2 | [Dual Goal Representations][p2] | 目标表示应保留什么信息 | 官方 PDF |
| P3 | [Demystifying The Mechanisms Behind Emergent Exploration in Goal-Conditioned RL][p3] | actor、critic 表示与数据收集的反馈关系 | [作者 arXiv v1][r3] |
| P4 | [Occupancy Reward Shaping: Improving Credit Assignment for Offline Goal-Conditioned Reinforcement Learning][p4] | 目标距离、学习信号与长时程目标到达 | [作者 arXiv v1][r4] |
| P5 | [Hierarchical Entity-centric Reinforcement Learning with Factored Subgoal Diffusion][p5] | 因子化表示、目标构造与子目标质量 | [作者 arXiv v1][r5] |
| P6 | [Stackelberg Coupling of Online Representation Learning and Reinforcement Learning][p6] | 表示学习和价值学习的分工 | 官方 PDF |
| P7 | [Ego-Foresight: Self-supervised Learning of Agent-Aware Representations for Improved RL][p7] | 分离表示的直观动机与经验验证 | [作者 arXiv v4][r7] |
| P8 | [Spectral Bellman Method: Unifying Representation and Exploration in RL][p8] | 表示学习与探索如何形成同一条贡献主线 | 官方 PDF |
| P9 | [Scalable Exploration for High-Dimensional Continuous Control via Value-Guided Flow][p9] | 在线价值梯度优化与更新步数分析 | [作者 arXiv v1][r9] |
| P10 | [Skill Learning via Policy Diversity Yields Identifiable Representations for Reinforcement Learning][p10] | 表示性质、理论主张和实验指标的对应关系 | 官方 PDF |

完整官方页面、PDF 地址和阅读版本保存在 `papers.json`。作者版与会议版可能有文字差异，本报告涉及作者版的章节与语言判断以链接版本为准。P6 使用会议版：它将 Q-function 设为慢速 leader、encoder 设为快速 follower，不能混用早期版本的角色描述。

**文件下载状态：**联网阅读工具可以访问上述正文，但当前服务器终端无法解析论文网站域名，原始 PDF 尚未下载到本地。本目录提供完整下载清单与脚本；阅读来源核实和本地 PDF 下载是分别记录的。

## 3. 逐篇分析

### P1. Multistep Quasimetric Distances：从已有优势收束到一个瓶颈

**叙事结构。** 摘要先建立 GCRL 与距离学习的联系，再指出现有拟度量方法在随机性、视觉输入和长时程中的困难，随后引入多步估计。正文依次展开背景、方法、分析、实验和结论。实验开头提出问题，后文回答泛化收益与设计选择。[摘要、§1、§4–6][p1]

**语言特点。** 作者用 “Our main contribution lies in” 直接定位贡献；技术细节围绕需要解决的瓶颈展开。方法部分也有 implementation 小节，但介绍的是算法如何运行，不是作者如何检查代码。

**对 NQRL 的启示。** 可以开门见山承接 QRL，再把论证收束到部分目标被补成全状态目标的问题。无需为了显得“原创”而隐藏继承关系。创新的清晰程度，来自问题与设计之间的对应。

### P2. Dual Goal Representations：先定义好表示的标准，再提出构造

**叙事结构。** 引言先问目标应如何表示，把要求凝练为信息充分性和对无关噪声的不变性，再介绍关系式目标表示。方法之后讨论相关工作，实验先做概念验证，再做基准与设计分析。[§1、§3–7][p2]

**语言特点。** 用 “Our key idea” 引出核心构造，随后分别报告理论与经验结果。值得注意的是，§6.2.2 明确讨论负面结果；积极的摘要与正文中诚实报告例外可以并存。

**对 NQRL 的启示。** 先交代两个角色：目标变量定义成功，其他变量提供控制所需的物理上下文。这样，分开编码就成为对任务结构的回应，而不只是网络模块介绍。该文的 dual 表示与 NQRL 的两类变量分离不是同一方法。

### P3. Emergent Exploration：现象、解释、可检验预测

**叙事结构。** 从一个已有算法的探索现象出发，提出 actor 与 critic 表示共同塑造探索的解释，再用简化模型、理论和干预实验检验。实验按照两个研究问题组织：表示如何演化，以及表示如何影响数据收集。[§1、§4–6][r3]

**语言特点。** 解释之后接具体预测，观察之后说明它回答了哪个问题。结尾集中交代研究范围和后续理论问题。

**对 NQRL 的启示。** 它是在线反馈叙事的近邻：访问行为影响训练数据，数据又改变表示。NQRL 可借鉴这一逻辑组织方式；但本篇的中心贡献就是解释机制，因此设置了专门的干预实验。NQRL 的辅助机制讨论不必照搬其全文规模，也不能把其结果当作 NQRL 自我修正已获验证的证据。

### P4. Occupancy Reward Shaping：将已有模型能力转化成学习信号

**叙事结构。** 从长时程信用分配困难出发，追问如何提取生成模型中的全局目标到达信息；方法按占用模型、几何信息、奖励和策略关系展开。实验用问题式小标题连接总体性能、价值学习与设计选择。[§1、§3–6][r4]

**语言特点。** 把核心收益放在主要结果中；对原因使用简短的 “We hypothesize”。baseline 来源在实验设置中直接交代，详细配置交给附录。

**对 NQRL 的启示。** 实验章节应回答方法带来的具体改变，而非逐张表复述数字。先报告成功率，再解释距离下降这一中间量提供了什么补充信息。baseline 的可信度可以通过来源、设置和适配范围简洁表达。

### P5. HECRL：两个设计原则如何共同解决一个问题

**叙事结构。** 将长时程困难与多实体组合复杂性联系起来，用层级结构和因子化结构共同回应。方法之后依次呈现性能、消融、子目标质量和泛化分析。[§1、§3–5][r5]

**语言特点。** 先解释一个模块为何必要，再介绍实现它的方式。§4.5 用假设引出子目标质量分析，说明作者可以主动提出机制解释，再选择有代表性的证据展开。

**对 NQRL 的启示。** 分开编码与潜空间补全应写成协同设计：前者固定目标语义，后者处理目标没有规定的上下文。消融的价值在于回答这两步分别解决什么，而非仅证明完整版本某个数字更大。

### P6. SCORER：通过明确分工讲清网络改动

**叙事结构。** 从表示学习与价值学习耦合造成的优化困难切入，提出角色分工，再落到两种时间尺度的更新。实验用对应消融考察角色、目标与更新安排。[摘要、§1、§4–6][p6]

**语言特点。** “Through this division of labor” 将结构安排直接连接到作用。它没有依靠罗列网络名称来解释贡献，而是强调各部分承担的任务。

**对 NQRL 的启示。** 解释 goal encoder 与 non-goal encoder 时，先写各自负责什么，再给公式。结构图应突出信息如何分流、保留和汇合；代码文件、函数名和检查过程不承担这一科学解释。

### P7. Ego-Foresight：用直观推理建立分离表示的动机

**叙事结构。** 先借助运动预测解释为何可以区分智能体与环境，再指出已有分离方法对监督信息的依赖，提出自监督方案。实验先看表示是否按设计分离，再看加入 RL 后的收益和消融。[§1、§3–4、§6][r7]

**语言特点。** 引言用 “Our insight is” 提出设计洞见，并用一段推理解释预期收益；不在每句机制解释后追加否定性声明。实验讨论中仍会说明具体方法在部分任务上的差异。

**对 NQRL 的启示。** 这是最值得参考的作者口吻之一：用真实状态例子把分离的理由讲透，再顺势提出设计。自我修正同样可以用一个完整、简短的反馈链解释，不必写成对他人产品的评估意见。

### P8. Spectral Bellman Method：理论视角贯穿实际方法

**叙事结构。** 把表示学习与探索放进统一问题，借助 Bellman 表示性质导出学习目标，再连接到 Q-learning 和探索策略。随后实验检验组合及其组成部分。[§1、§3–5、§7][p8]

**语言特点。** 数学性质出现时交代其成立条件；实际算法出现时说明怎样利用该性质。理论、设计和效果没有混成一个笼统的“更优”。

**对 NQRL 的启示。** 每个公式前先说明它要完成的任务。最近补全的优化式应接在“目标只规定部分变量”之后，让数学形式成为前面动机的自然表达。无需额外包装成尚未建立的理论保证。

### P9. Qflex：先诊断维度问题，再提出梯度引导

**叙事结构。** 在方法前单独诊断高维空间中无方向探索的困难，再提出价值引导流。实验由常规基准进入更高维控制，最后分析采样质量与超参数。[§4–7][r9]

**语言特点。** 主要结果聚焦性能和扩展能力；§6.3 把梯度步数、步长和求解步长作为敏感性分析，而不是要求这些参数与性能严格单调对应。

**对 NQRL 的启示。** 目标距离下降与最终任务成功应分别呈现。可以积极报告非零更新预算整体有效，而把预算选择放在消融中解释。两篇优化的变量不同：Qflex 引导动作，NQRL 优化非目标潜变量补全。

### P10. Identifiable Representations：让实验指标直接对应主张

**叙事结构。** 从相似技能学习方法为何表现不同的问题出发，以表示可辨识性建立分析视角，再讨论条件、方法启示和实验。实验同时考察表示性质与探索覆盖，避免单一指标产生误导。[§1、§4–6][p10]

**语言特点。** 有定理的地方明确报告证明，经验联系则作为经验结果表达。讨论部分解释分析所依赖的条件与适用范围。

**对 NQRL 的启示。** 先决定每个实验支持哪项主张：成功率支持控制收益，下降距离支持内部搜索行为。它们可以相互补充，但不能仅凭距离下降就宣称真实环境距离估计必然准确。

## 4. 共同叙事模式：值得借鉴的是论证次序

这些论文并不共享唯一的章节模板。按中心贡献，大致可以归纳为四种：

| 类型 | 论证次序 | 对应样本 | NQRL 的借鉴方式 |
| --- | --- | --- | --- |
| 具体瓶颈驱动的方法改进 | 已有价值 → 未解决的障碍 → 关键设计 → 实验收益 | P1、P4、P5、P6、P9 | 作为全文主线 |
| 表示设计驱动 | 好表示的要求 → 现有表示的不足 → 新构造 → 性质与效果 | P2、P7 | 用于动机和分离编码部分 |
| 机制解释驱动 | 现象 → 竞争解释 → 预测 → 干预检验 | P3 | 借鉴反馈逻辑，不将其完整范式套入全文 |
| 理论结构驱动 | 数学性质 → 条件与推导 → 方法启示 → 验证 | P8、P10 | 借鉴主张与证据的对齐 |

NQRL 目前最适合以第一类为骨架，以第二类解释核心设计。自我修正是在线训练流程的一段机制说明，不必为了它把全文改造成第三类论文。

### 摘要：用有限篇幅完成贡献叙事

这些样本的摘要把中心篇幅交给问题、洞见、方法和主要结果。NQRL 可以按下面的次序组织：

1. QRL 如何利用拟度量结构进行目标到达学习。
2. 在本文部分目标接口中，零填充将目标补成全状态输入，引出表示不匹配。
3. NQRL 分开编码，并在固定目标表示下优化非目标潜变量补全。
4. 用一句话交代 online interaction setting。
5. 用主实验与梯度更新消融的代表性数字收尾。

无需在摘要里插入“尽管单个任务不是单调改善”。如果没有声称单调性，就无需在这里主动反驳它。主结果应保持准确的比较范围，例如当前稿件中的“13/14 个任务优于 QRL、12/14 个任务在比较方法中均值最高”，而不是改成“所有任务都最优”。

### 引言：每段推进一个新的问题

建议保留四个推进步骤：部分目标的实际含义；QRL 所用接口的具体目标构造；NQRL 如何处理目标语义与控制上下文；实验回答了什么。状态例子要早于长公式，使读者先明白为何目标位置不应同时规定一组全零姿态和速度。

引言应承接已有方法，而不是泛泛宣称“现有强化学习不够好”。“我们改变了什么”需要比“这个方向很重要”获得更多篇幅。

### 方法：先说明设计理由，再给运算

NQRL 的方法可以始终围绕三个连续问题组织：

1. 哪些变量定义目标，哪些变量仍影响行动后果？对应 separate encoders。
2. 目标没有规定剩余变量时，critic 应评估什么？对应 conditional latent completion。
3. 新发生的环境交互如何继续影响学习？对应真实 transition、critic 更新与反馈解释。

架构图承担前两个问题的直观表达；结果表承担效果比较。P2、P6 的概念图值得借鉴，P4 也说明首图可以是聚合结果图。重点是每张图都有独立的信息价值，不必规定所有论文的 Figure 1 都必须画网络。

### 实验：先回答研究问题，再读数字

适合当前证据的组织方式是：整体比较是否有效；QRL 失败任务中是否有改善；目标距离更新是否带来收益；必要的其他消融与训练行为分析。每小节首句提出或回答一个问题，再指出支持它的结果。

对于用户提供的五任务、五 seed 更新预算消融，最直接的主结论是：15 个非零预算与任务组合的成功率均值均高于对应零更新版本；宏平均成功率从 0.34260 提升到 0.47320，即提高 13.06 个百分点。下降距离和相对下降百分比是搜索行为指标，应保留各自定义。

这些均值足以支持“综合来看有效”的经验表述。跨 seed 不确定性仍由原表呈现；没有逐 seed 配对数据和相应检验时，不把均值优势改写成统计显著性结论。当前梯度步数实验也不直接测量自我修正。

### 局限性：集中讨论整体边界，具体例外在结果中自然交代

P1、P4、P5、P7 等在结尾讨论整体局限；P2 在相关实验中明确讨论负面结果。可借鉴的不是“所有不利信息都移到最后”，而是避免在每段后重复同一句免责声明。

NQRL 可以在结尾集中讨论任务范围、部分目标划分的依赖、潜空间搜索的代价与未覆盖的设置。在实验正文中保留具体结果差异即可，无需反复回到“因此不能保证……”的句式。

## 5. 语言风格：怎样保持作者口吻

### 用动词标明句子的工作

| 句子要完成什么 | 合适的表达 | 写作重点 |
| --- | --- | --- |
| 提出方法 | We propose / We introduce / We design | 明确作者做了什么 |
| 描述确定的结构 | The goal encoder maps... / The search fixes... | 主语具体，避免抽象修饰词堆积 |
| 报告实验 | We observe / Our experiments show / achieves | 紧接指标、比较对象与范围 |
| 解释机制 | This design allows... / This feedback provides... | 给出中间因果环节 |
| 提出尚未独立识别的原因 | We hypothesize / We attribute this to... | 一次标明解释性质，随后把理由讲清楚 |
| 报告证明 | We prove / Under the stated assumptions... | 仅用于实际建立的理论结果 |

这些是写作功能的归纳，不是必须逐句套用的模板。尤其不建议为了显得像作者，就机械地给每一句加上 “we”。方法、模型、数据和反馈本身也可以作主语。

### 自我修正段落的建议写法

下面是为 NQRL 原创的写作示例，不是所读论文的原文，也不表示新增了机制实验：

> In our online setting, distance estimates also influence where new data are collected. If the critic assigns an erroneously low distance to a reachable region, the actor is encouraged to visit that region. The resulting real transitions are then used to update the critic, providing fresh evidence about its dynamics and goal-reaching distance. This feedback gives NQRL a pathway to progressively correct inaccurate estimates through interaction.

对应中文：

> 在本文的在线交互设置中，距离估计也影响新数据的采集位置。当 critic 错误地低估某个可达区域的目标距离时，actor 会更倾向于访问该区域。由此产生的真实 transition 随后用于更新 critic，为该区域的动力学与目标到达距离提供新的学习依据。这一反馈使 NQRL 能够通过交互逐步修正不准确的估计。

这里的主体始终是我们的方法及其运行过程。低估、访问、采样和更新之间的联系已经讲清楚，无需紧接着追加两句否定性评价。这里解释的是修正途径；“新样本必然使所有估计恢复准确”不是该推理已经建立的命题。

### 将工程过程改写为科学信息

论文需要实现来源、协议和关键参数，而不需要作者检查这些信息的工作日记。baseline 段落直接交代公开实现的移植来源、必要的容量与环境适配，以及实际验证范围即可。容量或训练过程存在实质差别时，应直接披露差别。

同理，数据来自另一台服务器不影响它成为正式实验结果。主文写训练预算、任务、seed 数量、指标和聚合方式；机器之间如何传输日志属于研究记录。已观察到的结果、正在进行的验证和未来工作，在成稿时应各用对应的时态。

## 6. 对当前 NQRL 稿件的具体判断

当前英文摘要已经采用“QRL → 部分目标不匹配 → NQRL → 在线实验 → 性能收益”的结构，与本次最相关的方法论文相符。真正值得继续优化的不是添加更多防御性限定，而是以下四点：

1. 进一步把“为什么分开”与“为什么还要补全”写成连续论证，减少公式之间重复描述实现的文字。
2. 对 QRL 的问题保持精确归属：当前接口中零填充的目标命令，与训练时使用的 replay/hindsight 全状态目标产生不匹配。不要把它扩大为所有 QRL 训练目标均由零填充生成。
3. 将自我修正保持为一段作者视角的在线反馈解释；主实验继续承担性能证据，梯度消融承担搜索收益证据。
4. 在相关工作中考虑加入 P1、P2 和 P3，分别讨论拟度量估计、目标表示和交互反馈。P1、P2 主要属于离线设置，不能把它们的实验结论自动转移到 NQRL 的在线实验中。

最优先精读的顺序是 P1、P2、P7、P3、P9：依次对应全文主线、表示动机、作者口吻、反馈解释和梯度消融。其余五篇有助于检查方法分工、指标选择及实验组织。

本次任务仅生成阅读分析与下载材料，没有修改论文正文或投稿 ZIP。上述组织建议也不是 ICLR 的强制模板；会议格式和篇幅限制仍需以实际投稿年份的要求为准。

[p1]: https://proceedings.iclr.cc/paper_files/paper/2026/file/eddc0fab5a42f8d8de6eb5566cd9f1d3-Paper-Conference.pdf
[p2]: https://proceedings.iclr.cc/paper_files/paper/2026/file/f8cd7eb180deb5c6ec876de2b442fa72-Paper-Conference.pdf
[p3]: https://proceedings.iclr.cc/paper_files/paper/2026/hash/cbaffeeda13dbd8bf9489feb3f198ff4-Abstract-Conference.html
[p4]: https://proceedings.iclr.cc/paper_files/paper/2026/hash/5683583b88da51c79d2eb263f295286e-Abstract-Conference.html
[p5]: https://proceedings.iclr.cc/paper_files/paper/2026/hash/de2bc3eed995c24307e8011161ac1438-Abstract-Conference.html
[p6]: https://proceedings.iclr.cc/paper_files/paper/2026/file/050f729db878738a9d44d4a7f995bf36-Paper-Conference.pdf
[p7]: https://proceedings.iclr.cc/paper_files/paper/2026/hash/7504142a20a3e1fe9dd7de42f475828c-Abstract-Conference.html
[p8]: https://proceedings.iclr.cc/paper_files/paper/2026/file/58a5aafe8df63a20a7f6d7840100067c-Paper-Conference.pdf
[p9]: https://proceedings.iclr.cc/paper_files/paper/2026/hash/5a05d477fef5a43c805d3377700efa40-Abstract-Conference.html
[p10]: https://proceedings.iclr.cc/paper_files/paper/2026/file/247a729f6b08f91a9b27de6d8e38980b-Paper-Conference.pdf
[r3]: https://arxiv.org/html/2510.14129v1
[r4]: https://arxiv.org/html/2604.20627v1
[r5]: https://arxiv.org/html/2602.02722v1
[r7]: https://arxiv.org/html/2407.01570v4
[r9]: https://arxiv.org/html/2601.19707v1
