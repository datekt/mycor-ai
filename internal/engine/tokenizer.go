package engine

import (
	"strings"
	"unicode/utf8"
)

// emoticonList is read-only after initialisation and therefore needs no lock.
var emoticonList = []string{
	":-)", ":-(", ":-d", ":-p", ":-/", ":-*", ":-'(",
	":')", ":d", ":p", ":3", ":o", ":/", ":|", ":'(",
	":)", ":(", ";-)", ";)", ";(", ":*", "^^", "<3", "o_o", "-_-",
}

func matchEmoticon(s string, pos int) int {
	remaining := s[pos:]
	best := 0
	for _, e := range emoticonList {
		if len(e) > best && strings.HasPrefix(remaining, e) {
			best = len(e)
		}
	}
	return best
}

func isPunctByte(c byte) bool {
	switch c {
	case '.', ',', '?', '!', ':', ';':
		return true
	}
	return false
}

func isPunctRune(r rune) bool {
	switch r {
	case '.', ',', '?', '!', ':', ';':
		return true
	case '—', '–', '«', '»', '…', '“', '”', '‘', '’', '‒', '―':
		return true
	}
	return false
}

// Tokenize splits text into lowercased words, punctuation and emoticons. It is
// a pure function: no state is read or written.
func Tokenize(text string) []string {
	lower := strings.ToLower(text)
	tokens := make([]string, 0, 16)
	var buf strings.Builder

	i := 0
	for i < len(lower) {
		c := lower[i]
		if c < 128 {
			if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
				if buf.Len() > 0 {
					tokens = append(tokens, buf.String())
					buf.Reset()
				}
				i++
				continue
			}
			if length := matchEmoticon(lower, i); length > 0 {
				if buf.Len() > 0 {
					tokens = append(tokens, buf.String())
					buf.Reset()
				}
				tokens = append(tokens, lower[i:i+length])
				i += length
				continue
			}
			if isPunctByte(c) {
				if buf.Len() > 0 {
					tokens = append(tokens, buf.String())
					buf.Reset()
				}
				tokens = append(tokens, string(c))
				i++
				continue
			}
			buf.WriteByte(c)
			i++
			continue
		}

		r, size := utf8.DecodeRuneInString(lower[i:])
		if r == utf8.RuneError && size == 1 {
			buf.WriteByte(c)
			i++
			continue
		}
		if isPunctRune(r) {
			if buf.Len() > 0 {
				tokens = append(tokens, buf.String())
				buf.Reset()
			}
			tokens = append(tokens, string(r))
			i += size
			continue
		}
		buf.WriteRune(r)
		i += size
	}
	if buf.Len() > 0 {
		tokens = append(tokens, buf.String())
	}
	return tokens
}

func isPunct(w string) bool {
	switch w {
	case ".", ",", "?", "!", ":", ";", "…":
		return true
	}
	return false
}

// JoinWords renders tokens back into readable text, suppressing the space
// before punctuation. It is a pure function.
func JoinWords(words []string) string {
	var sb strings.Builder
	for i, w := range words {
		if i > 0 && !isPunct(w) {
			sb.WriteByte(' ')
		}
		sb.WriteString(w)
	}
	return sb.String()
}
