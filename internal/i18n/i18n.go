package i18n

type Lang string

const (
	EN Lang = "en"
	RU Lang = "ru"
)

type Messages struct {
	YouRole          string
	AIRole           string
	TeacherRole      string
	YouPrompt        string
	AI               string
	Thoughts         string
	IdleThought      string
	CorrectAnswer    string
	BrainLoaded      string
	BrainEmpty       string
	BrainLoadedInfo  string
	ThinkingEnabled  string
	SessionReset     string
	HelpHeader       string
	HelpHelp         string
	HelpStats        string
	HelpHistory      string
	HelpUndo         string
	HelpReset        string
	HelpTemp         string
	HelpLR           string
	HelpMomentum     string
	HelpImport       string
	HelpSelf         string
	HelpLang         string
	HelpIdle         string
	HelpExit         string
	HelpThinkingNote string
	HistoryEmpty     string
	HistoryHeader    string
	UndoOK           string
	UndoFail         string
	ResetOK          string
	ResetFail        string
	TempRange        string
	TempOK           string
	LRRange          string
	LROK             string
	MomentumRange    string
	MomentumOK       string
	ImportReading    string
	ImportOK         string
	ImportFail       string
	ImportUsage      string
	StatsHeader      string
	StatsThinking    string
	StatsNoThinking  string
	StatsRecent      string
	StatsNoRecent    string
	Synapse          string
	SelfHeader       string
	SelfMirror       string
	SelfSeed         string
	SaveFail         string
	LangChanged      string
	LangUsage        string
	IdleOn           string
	IdleOff          string
	IdleUsage        string
	LossLine         string
	LangSelect       string
	LangChoice       string
}

var current = EN

func Init(l Lang) {
	current = l
}

func Set(l Lang) {
	current = l
}

func Current() Lang {
	return current
}

func M() Messages {
	if current == RU {
		return ru
	}
	return en
}
