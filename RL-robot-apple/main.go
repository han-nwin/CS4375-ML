package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// Cell types
const (
	EMPTY = "."
	WALL  = "#"
	GOOD  = "G"
	BAD   = "B"
)

// Actions
var ACTIONS = []string{"UP", "DOWN", "LEFT", "RIGHT", "STAY"}

// Direction vectors
var DIRS = map[string][2]int{
	"UP":    {-1, 0},
	"DOWN":  {1, 0},
	"LEFT":  {0, -1},
	"RIGHT": {0, 1},
	"STAY":  {0, 0},
}

// Arrow symbols
var ARROWS = map[string]string{
	"UP":    "↑",
	"DOWN":  "↓",
	"LEFT":  "←",
	"RIGHT": "→",
}

// Styles
var (
	wallStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("7"))
	robotStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("11")).Bold(true)
	goodStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("10")).Bold(true)
	badStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("9")).Bold(true)
	emptyStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("8"))
	headerStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("12"))
	infoStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("7"))
)

// Apple represents an apple with its type and TTL
type Apple struct {
	Kind string
	TTL  int
}

// State represents the observation state
type State struct {
	Cur   string
	Up    string
	Down  string
	Left  string
	Right string
}

// AppleWorld represents the grid world environment
type AppleWorld struct {
	h         int
	w         int
	walls     [][]bool
	apples    map[[2]int]Apple
	robot     [2]int
	robotDir  string
	t         int
	goodEaten int
	badEaten  int
	rng       *rand.Rand

	// Config
	spawnGood [2]int
	spawnBad  [2]int
	appleTTL  [2]int
}

// NewAppleWorld creates a new AppleWorld
func NewAppleWorld(h, w int, wallProb float64, seed int64) *AppleWorld {
	aw := &AppleWorld{
		h:         h,
		w:         w,
		walls:     make([][]bool, h),
		apples:    make(map[[2]int]Apple),
		robotDir:  "RIGHT",
		rng:       rand.New(rand.NewSource(seed)),
		spawnGood: [2]int{1, 3},
		spawnBad:  [2]int{1, 2},
		appleTTL:  [2]int{8, 16},
	}

	// Initialize walls
	for r := 0; r < h; r++ {
		aw.walls[r] = make([]bool, w)
		for c := 0; c < w; c++ {
			// Border walls
			if r == 0 || c == 0 || r == h-1 || c == w-1 {
				aw.walls[r][c] = true
			} else {
				aw.walls[r][c] = aw.rng.Float64() < wallProb
			}
		}
	}

	// Ensure roomy center
	for r := 1; r < h-1; r++ {
		for c := 1; c < w-1; c++ {
			if (r == h/2 || r == h/2-1) && (c == w/2 || c == w/2-1) {
				aw.walls[r][c] = false
			}
		}
	}

	aw.Reset()
	return aw
}

// Reset resets the environment
func (aw *AppleWorld) Reset() State {
	aw.apples = make(map[[2]int]Apple)
	aw.t = 0
	aw.goodEaten = 0
	aw.badEaten = 0
	aw.robot = aw.randomEmptyCell()
	aw.robotDir = "RIGHT"
	return aw.Observe()
}

// randomEmptyCell finds a random empty cell
func (aw *AppleWorld) randomEmptyCell() [2]int {
	for {
		r := aw.rng.Intn(aw.h-2) + 1
		c := aw.rng.Intn(aw.w-2) + 1
		pos := [2]int{r, c}
		if !aw.walls[r][c] {
			if _, exists := aw.apples[pos]; !exists {
				return pos
			}
		}
	}
}

// spawnApples spawns new apples
func (aw *AppleWorld) spawnApples() {
	g := aw.rng.Intn(aw.spawnGood[1]-aw.spawnGood[0]+1) + aw.spawnGood[0]
	b := aw.rng.Intn(aw.spawnBad[1]-aw.spawnBad[0]+1) + aw.spawnBad[0]

	for i := 0; i < g; i++ {
		aw.spawnOne(GOOD)
	}
	for i := 0; i < b; i++ {
		aw.spawnOne(BAD)
	}
}

// spawnOne spawns a single apple
func (aw *AppleWorld) spawnOne(kind string) {
	for i := 0; i < 20; i++ {
		pos := aw.randomEmptyCell()
		if pos == aw.robot {
			continue
		}
		ttl := aw.rng.Intn(aw.appleTTL[1]-aw.appleTTL[0]+1) + aw.appleTTL[0]
		aw.apples[pos] = Apple{Kind: kind, TTL: ttl}
		return
	}
}

// decayApples decreases TTL and removes expired apples
func (aw *AppleWorld) decayApples() {
	toDel := [][2]int{}
	for pos, apple := range aw.apples {
		apple.TTL--
		if apple.TTL <= 0 {
			toDel = append(toDel, pos)
		} else {
			aw.apples[pos] = apple
		}
	}
	for _, pos := range toDel {
		delete(aw.apples, pos)
	}
}

// cellType returns the type of cell at (r, c)
func (aw *AppleWorld) cellType(r, c int) string {
	if aw.walls[r][c] {
		return WALL
	}
	pos := [2]int{r, c}
	if apple, exists := aw.apples[pos]; exists {
		return apple.Kind
	}
	return EMPTY
}

// Observe returns the current observation state
func (aw *AppleWorld) Observe() State {
	r, c := aw.robot[0], aw.robot[1]
	return State{
		Cur:   aw.cellType(r, c),
		Up:    aw.cellType(r-1, c),
		Down:  aw.cellType(r+1, c),
		Left:  aw.cellType(r, c-1),
		Right: aw.cellType(r, c+1),
	}
}

// Step performs one action and returns next state, reward, done
func (aw *AppleWorld) Step(action string) (State, float64, bool) {
	aw.t++
	reward := -0.05

	r, c := aw.robot[0], aw.robot[1]

	if dir, exists := DIRS[action]; exists {
		nr, nc := r+dir[0], c+dir[1]
		if aw.walls[nr][nc] {
			reward -= 0.75
		} else {
			aw.robot = [2]int{nr, nc}
			if action != "STAY" {
				aw.robotDir = action
			}

			// Auto-eat apple
			if apple, exists := aw.apples[aw.robot]; exists {
				delete(aw.apples, aw.robot)
				if apple.Kind == GOOD {
					reward += 10.0
					aw.goodEaten++
				} else {
					reward -= 10.0
					aw.badEaten++
				}
			}
		}
	}

	aw.decayApples()
	aw.spawnApples()

	return aw.Observe(), reward, false
}

// Render returns a string representation of the world
func (aw *AppleWorld) Render() string {
	var sb strings.Builder
	rr, rc := aw.robot[0], aw.robot[1]
	robotSymbol := ARROWS[aw.robotDir]

	for r := 0; r < aw.h; r++ {
		for c := 0; c < aw.w; c++ {
			if aw.walls[r][c] {
				sb.WriteString(wallStyle.Render("#"))
			} else if r == rr && c == rc {
				sb.WriteString(robotStyle.Render(robotSymbol))
			} else if apple, exists := aw.apples[[2]int{r, c}]; exists {
				if apple.Kind == GOOD {
					sb.WriteString(goodStyle.Render("G"))
				} else {
					sb.WriteString(badStyle.Render("B"))
				}
			} else {
				sb.WriteString(emptyStyle.Render("."))
			}
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

// QAgent implements Q-Learning
type QAgent struct {
	actions    []string
	Q          map[State]map[string]float64
	alpha      float64
	gamma      float64
	epsilon    float64
	epsDecay   float64
	epsMin     float64
	rng        *rand.Rand
	lastAction string
	lastReward float64
}

// NewQAgent creates a new Q-learning agent
func NewQAgent(actions []string, alpha, gamma, epsilon, epsDecay, epsMin float64) *QAgent {
	return &QAgent{
		actions:  actions,
		Q:        make(map[State]map[string]float64),
		alpha:    alpha,
		gamma:    gamma,
		epsilon:  epsilon,
		epsDecay: epsDecay,
		epsMin:   epsMin,
		rng:      rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// Act selects an action using epsilon-greedy policy
func (qa *QAgent) Act(state State) string {
	if qa.rng.Float64() < qa.epsilon {
		return qa.actions[qa.rng.Intn(len(qa.actions))]
	}

	if _, exists := qa.Q[state]; !exists {
		qa.Q[state] = make(map[string]float64)
		for _, a := range qa.actions {
			qa.Q[state][a] = 0.0
		}
	}

	bestAction := qa.actions[0]
	bestValue := qa.Q[state][bestAction]
	for _, a := range qa.actions {
		if qa.Q[state][a] > bestValue {
			bestValue = qa.Q[state][a]
			bestAction = a
		}
	}
	return bestAction
}

// Update performs Q-learning update
func (qa *QAgent) Update(s State, a string, r float64, s2 State) {
	if _, exists := qa.Q[s]; !exists {
		qa.Q[s] = make(map[string]float64)
		for _, act := range qa.actions {
			qa.Q[s][act] = 0.0
		}
	}
	if _, exists := qa.Q[s2]; !exists {
		qa.Q[s2] = make(map[string]float64)
		for _, act := range qa.actions {
			qa.Q[s2][act] = 0.0
		}
	}

	qSA := qa.Q[s][a]
	bestNext := -math.Inf(1)
	for _, v := range qa.Q[s2] {
		if v > bestNext {
			bestNext = v
		}
	}
	target := r + qa.gamma*bestNext
	qa.Q[s][a] = (1-qa.alpha)*qSA + qa.alpha*target

	qa.lastAction = a
	qa.lastReward = r
}

// DecayEpsilon decays epsilon
func (qa *QAgent) DecayEpsilon() {
	qa.epsilon = math.Max(qa.epsMin, qa.epsilon*qa.epsDecay)
}

// TickMsg is sent on each tick
type TickMsg time.Time

// Model is the bubbletea model
type model struct {
	env          *AppleWorld
	agent        *QAgent
	state        State
	totalReward  float64
	step         int
	maxSteps     int
	running      bool
	paused       bool
	tickInterval time.Duration
}

// initialModel creates the initial model
func initialModel() model {
	env := NewAppleWorld(25, 50, 0.12, 0)
	agent := NewQAgent(ACTIONS, 0.25, 0.90, 0.35, 0.9995, 0.05)
	state := env.Reset()

	return model{
		env:          env,
		agent:        agent,
		state:        state,
		totalReward:  0.0,
		step:         0,
		maxSteps:     6000,
		running:      false,
		paused:       false,
		tickInterval: 50 * time.Millisecond,
	}
}

// Init initializes the model
func (m model) Init() tea.Cmd {
	return nil
}

// Update updates the model
func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch msg.String() {
		case "ctrl+c", "q":
			return m, tea.Quit
		case " ":
			if !m.running {
				m.running = true
				return m, tick(m.tickInterval)
			} else {
				m.paused = !m.paused
				if !m.paused {
					return m, tick(m.tickInterval)
				}
			}
		case "r":
			m.state = m.env.Reset()
			m.agent = NewQAgent(ACTIONS, 0.25, 0.90, 0.35, 0.9995, 0.05)
			m.totalReward = 0.0
			m.step = 0
			m.running = false
			m.paused = false
		case "+", "=":
			if m.tickInterval > 10*time.Millisecond {
				m.tickInterval -= 10 * time.Millisecond
			}
		case "-", "_":
			if m.tickInterval < 500*time.Millisecond {
				m.tickInterval += 10 * time.Millisecond
			}
		}

	case TickMsg:
		if m.running && !m.paused && m.step < m.maxSteps {
			// Perform one step
			action := m.agent.Act(m.state)
			nextState, reward, _ := m.env.Step(action)
			m.agent.Update(m.state, action, reward, nextState)
			m.agent.DecayEpsilon()

			m.state = nextState
			m.totalReward += reward
			m.step++

			return m, tick(m.tickInterval)
		}
	}

	return m, nil
}

// View renders the view
func (m model) View() string {
	var sb strings.Builder

	// Header
	sb.WriteString(headerStyle.Render("=== Q-Learning Robot Apple Collector ==="))
	sb.WriteString("\n\n")

	// Grid
	sb.WriteString(m.env.Render())
	sb.WriteString("\n")

	// Stats
	sb.WriteString(infoStyle.Render(fmt.Sprintf("Step: %d/%d  ", m.step, m.maxSteps)))
	sb.WriteString(infoStyle.Render(fmt.Sprintf("Total Reward: %.1f  ", m.totalReward)))
	sb.WriteString(infoStyle.Render(fmt.Sprintf("Epsilon: %.3f\n", m.agent.epsilon)))

	sb.WriteString(infoStyle.Render(fmt.Sprintf("Last Action: %s  ", m.agent.lastAction)))
	sb.WriteString(infoStyle.Render(fmt.Sprintf("Last Reward: %+.2f\n", m.agent.lastReward)))

	sb.WriteString(goodStyle.Render(fmt.Sprintf("Good Apples: %d", m.env.goodEaten)))
	sb.WriteString("  ")
	sb.WriteString(badStyle.Render(fmt.Sprintf("Bad Apples: %d", m.env.badEaten)))
	sb.WriteString("\n\n")

	// Legend
	sb.WriteString(goodStyle.Render("G") + " = Good Apple (+10)  ")
	sb.WriteString(badStyle.Render("B") + " = Bad Apple (-10)  ")
	sb.WriteString(robotStyle.Render("↑↓←→") + " = Robot\n\n")

	// Controls
	sb.WriteString(headerStyle.Render("Controls:") + "\n")
	sb.WriteString("  SPACE - Start/Pause\n")
	sb.WriteString("  R     - Reset\n")
	sb.WriteString("  +/-   - Speed Up/Down\n")
	sb.WriteString("  Q     - Quit\n\n")

	// Status
	if !m.running {
		sb.WriteString(infoStyle.Render("Press SPACE to start training"))
	} else if m.paused {
		sb.WriteString(infoStyle.Render("PAUSED - Press SPACE to resume"))
	} else if m.step >= m.maxSteps {
		sb.WriteString(headerStyle.Render("Training complete!"))
	} else {
		sb.WriteString(infoStyle.Render("Training..."))
	}

	return sb.String()
}

// tick creates a tick command
func tick(d time.Duration) tea.Cmd {
	return tea.Tick(d, func(t time.Time) tea.Msg {
		return TickMsg(t)
	})
}

func main() {
	p := tea.NewProgram(initialModel(), tea.WithAltScreen())
	if _, err := p.Run(); err != nil {
		fmt.Printf("Error: %v", err)
		os.Exit(1)
	}
}
