import java.util.*;
public class GrammarSolver {
    TreeMap<String, ArrayList> grammarMap;
    // store the grammar in a TreeMap. Nonterminal symbol as the key, rules as the values
    public GrammarSolver(List<String> grammar) {
         grammarMap = new TreeMap<>();
         if (!grammar.isEmpty()) {
             for (String eachGrammar : grammar){
                 ArrayList<String[]> processedElements = new ArrayList<>();
                 String[] grammarPair = eachGrammar.split("::=");
                 String[] elements = grammarPair[1].split("[|]+");
                 //grammarMap.put(grammarPair[0], elements);
                 for (int i = 0; i < elements.length; i++) {
                     if (elements[i].contains(" ")) {
                         processedElements.add(elements[i].split(" "));
                     } else {
                         processedElements.add(new String[] {elements[i]});
                     }
                 }
                 grammarMap.put(grammarPair[0], processedElements);
             }
         } else {
             throw new IllegalArgumentException("grammar can not be empty");
         }
    }
    // return true if the given symbol is a nonterminal of the grammar
    public boolean grammarContains(String symbol) {
        return grammarMap.containsKey(symbol);
    }
    //  use the grammar to randomly generate the given number of occurrences of
    // the given symbol and return the results as an array of strings
    public String[] generate(String symbol, int times) {
        ArrayList<List<String>> final_result = new ArrayList<>();
        Random rand = new Random();
        for (int i = 0; i < times; i++) {
            // For example
            // availableSentenceStructures = [[<dp>, <adjp>, <n>], [<pn>]]
            ArrayList<String[]> availableSentenceStructures = grammarMap.get(symbol);
            int randomIndex = rand.nextInt(availableSentenceStructures.size());
            // For example
            // sentenceConstructor = [<dp>, <adjp>, <n>]
            String[] sentenceConstructor = availableSentenceStructures.get(randomIndex);
            List<String> sentence = new ArrayList<>();
            for (String wordLookup:sentenceConstructor) { // wordLookup = <dp>
                String[] word;
                // wordLookup is a grammar key. wordLookup = <dp>
                if (grammarMap.containsKey(wordLookup)) {
                    // call recursive


                    word = generate(wordLookup, 1);
                } else {
                    // wordLookup is a not grammar key but a word.
                    // For example, wordLookup = "the"
                    word = new String[] {wordLookup};
                }
                // Note word = [SOMEWORD] is an array
                // For example, word = ["the"]
                sentence.add(word[0]); // word[0] is "the"
            }

            // Note that sentence is an ArrayList.
            // For example, [a, subliminal, mother]
            final_result.add(sentence);

        }

        String[] final_array = new String[final_result.size()];
        for (int i = 0; i < final_array.length; i++) {
            // Note that sentence is an ArrayList.
            // For example, [a, subliminal, mother]
            // We need to remove [,] to make it a pretty single string
            // such as "a subliminal mother"
            final_array[i] = final_result.get(i).toString()
                    .replace(",", "")  //remove the commas
                    .replace("[", "")  //remove the right bracket
                    .replace("]", "")  //remove the left bracket
                    .trim();
        }
        return final_array;

    }

    // return a string representation of the various nonterminal symbols (sorted)
    public String getSymbols() {
        return grammarMap.keySet().toString();
    }

}
