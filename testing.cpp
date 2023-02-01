#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <set>
//#include "search_server.h"
using namespace std;

const int MAX_RESULT_DOCUMENT_COUNT = 5;

string ReadLine() {
    string s;
    getline(cin, s);
    return s;
}

int ReadLineWithNumber() {
    int result;
    cin >> result;
    ReadLine();
    return result;
    
}

vector<string> SplitIntoWords(const string& text) {
    vector<string> words;
    string word;
    for (const char c : text) {
        if (c == ' ') {
            if (!word.empty()) {
                words.push_back(word);
                word.clear();
            }
        } else {
            word += c;
        }
    }
    if (!word.empty()) {
        words.push_back(word);
    }
    
    return words;
}

struct Document {
    int id;
    double relevance;
    int rating;
};

enum class DocumentStatus {
    ACTUAL,
    IRRELEVANT,
    BANNED,
    REMOVED,
};

class SearchServer {
public:
    void SetStopWords(const string& text) {
        for (const string& word : SplitIntoWords(text)) {
            stop_words_.insert(word);
        }
    }
    
    void AddDocument(int document_id, const string& document, DocumentStatus status,
                     const vector<int>& ratings) {
        const vector<string> words = SplitIntoWordsNoStop(document);
        const double inv_word_count = 1.0 / words.size();
        for (const string& word : words) {
            word_to_document_freqs_[word][document_id] += inv_word_count;
        }
        documents_.emplace(document_id, DocumentData{ComputeAverageRating(ratings), status});
    }
    
    vector<Document> FindTopDocuments(const string& raw_query) const {
        return FindTopDocuments(raw_query, DocumentStatus::ACTUAL);
    }
    
    vector<Document> FindTopDocuments(const string& raw_query, DocumentStatus status0) const {
        return FindTopDocuments(raw_query, [status0](int id, DocumentStatus status, int rating){
            return status0 == status;
        });
    }
    
    template <typename func>
    vector<Document> FindTopDocuments(const string& raw_query, func filter = [](int document_id, DocumentStatus status, int rating) { return status == DocumentStatus::ACTUAL; }) const {
        const Query query = ParseQuery(raw_query);
        vector<Document> matched_documents;
        for(const Document& doc : FindAllDocuments(query, filter)){
            if(filter(doc.id, documents_.at(doc.id).status, doc.rating))
                matched_documents.push_back(doc);
        }
        sort(matched_documents.begin(), matched_documents.end(),
             [](const Document& lhs, const Document& rhs) {
            if (abs(lhs.relevance - rhs.relevance) < 1e-6) {
                return lhs.rating > rhs.rating;
            } else {
                return lhs.relevance > rhs.relevance;
            }
        });
        if (matched_documents.size() > MAX_RESULT_DOCUMENT_COUNT) {
            matched_documents.resize(MAX_RESULT_DOCUMENT_COUNT);
        }
        return matched_documents;
    }
    
    int GetDocumentCount() const {
        return (int)documents_.size();
    }
    
    tuple<vector<string>, DocumentStatus> MatchDocument(const string& raw_query,
                                                        int document_id) const {
        const Query query = ParseQuery(raw_query);
        vector<string> matched_words;
        for (const string& word : query.plus_words) {
            if (word_to_document_freqs_.count(word) == 0) {
                continue;
            }
            if (word_to_document_freqs_.at(word).count(document_id)) {
                matched_words.push_back(word);
            }
        }
        for (const string& word : query.minus_words) {
            if (word_to_document_freqs_.count(word) == 0) {
                continue;
            }
            if (word_to_document_freqs_.at(word).count(document_id)) {
                matched_words.clear();
                break;
            }
        }
        return {matched_words, documents_.at(document_id).status};
    }
    
private:
    struct DocumentData {
        int rating;
        DocumentStatus status;
    };
    
    set<string> stop_words_;
    map<string, map<int, double>> word_to_document_freqs_;
    map<int, DocumentData> documents_;
    
    bool IsStopWord(const string& word) const {
        return stop_words_.count(word) > 0;
    }
    
    vector<string> SplitIntoWordsNoStop(const string& text) const {
        vector<string> words;
        for (const string& word : SplitIntoWords(text)) {
            if (!IsStopWord(word)) {
                words.push_back(word);
            }
        }
        return words;
    }
    
    static int ComputeAverageRating(const vector<int>& ratings) {
        if (ratings.empty()) {
            return 0;
        }
        int rating_sum = 0;
        for (const int rating : ratings) {
            rating_sum += rating;
        }
        return rating_sum / static_cast<int>(ratings.size());
    }
    
    struct QueryWord {
        string data;
        bool is_minus;
        bool is_stop;
    };
    
    QueryWord ParseQueryWord(string text) const {
        bool is_minus = false;
        // Word shouldn't be empty
        if (text[0] == '-') {
            is_minus = true;
            text = text.substr(1);
        }
        return {text, is_minus, IsStopWord(text)};
    }
    
    struct Query {
        set<string> plus_words;
        set<string> minus_words;
    };
    
    Query ParseQuery(const string& text) const {
        Query query;
        for (const string& word : SplitIntoWords(text)) {
            const QueryWord query_word = ParseQueryWord(word);
            if (!query_word.is_stop) {
                if (query_word.is_minus) {
                    query.minus_words.insert(query_word.data);
                } else {
                    query.plus_words.insert(query_word.data);
                }
            }
        }
        return query;
    }
    
    double ComputeWordInverseDocumentFreq(const string& word) const {
        return log(GetDocumentCount() * 1.0 / word_to_document_freqs_.at(word).size());
    }
    
    template <typename func>
    vector<Document> FindAllDocuments(const Query& query, func filter) const {
        map<int, double> document_to_relevance;
        for (const string& word : query.plus_words) {
            if (word_to_document_freqs_.count(word) == 0) {
                continue;
            }
            const double inverse_document_freq = ComputeWordInverseDocumentFreq(word);
            for (const auto [document_id, term_freq] : word_to_document_freqs_.at(word)) {
                if(documents_.count(document_id) && filter(document_id, documents_.at(document_id).status, documents_.at(document_id).rating))
                    document_to_relevance[document_id] += term_freq * inverse_document_freq;
            }
        }
        
        for (const string& word : query.minus_words) {
            if (word_to_document_freqs_.count(word) == 0) {
                continue;
            }
            for (const auto [document_id, _] : word_to_document_freqs_.at(word)) {
                document_to_relevance.erase(document_id);
            }
        }
        
        vector<Document> matched_documents;
        for (const auto [document_id, relevance] : document_to_relevance) {
            matched_documents.push_back(
                                        {document_id, relevance, documents_.at(document_id).rating});
        }
        return matched_documents;
    }
};

template <typename T, typename U>
void AssertEqualImpl(const T& t, const U& u, const string& t_str, const string& u_str, const string& file,
                     const string& func, unsigned line, const string& hint) {
    if (t != u) {
        cout << boolalpha;
        cout << file << "("s << line << "): "s << func << ": "s;
        cout << "ASSERT_EQUAL("s << t_str << ", "s << u_str << ") failed: "s;
        cout << t << " != "s << u << "."s;
        if (!hint.empty()) {
            cout << " Hint: "s << hint;
        }
        cout << endl;
        abort();
    }
}
#define ASSERT_EQUAL(a, b) AssertEqualImpl((a), (b), #a, #b, __FILE__, __FUNCTION__, __LINE__, ""s)
#define ASSERT_EQUAL_HINT(a, b, hint) AssertEqualImpl((a), (b), #a, #b, __FILE__, __FUNCTION__, __LINE__, (hint))

void AssertImpl(bool value, const string& expr_str, const string& file, const string& func, unsigned line,
                const string& hint) {
    if (!value) {
        cout << file << "("s << line << "): "s << func << ": "s;
        cout << "ASSERT("s << expr_str << ") failed."s;
        if (!hint.empty()) {
            cout << " Hint: "s << hint;
        }
        cout << endl;
        abort();
    }
}
#define ASSERT(expr) AssertImpl(!!(expr), #expr, __FILE__, __FUNCTION__, __LINE__, ""s)
#define ASSERT_HINT(expr, hint) AssertImpl(!!(expr), #expr, __FILE__, __FUNCTION__, __LINE__, (hint))

template <typename F>
void RunTestImpl(F func, const string& func_name) {
    func();
    cerr << func_name << " OK"s << endl;
}

#define RUN_TEST(func) RunTestImpl(func, #func)


// -------- Начало модульных тестов поисковой системы ----------

// Тест проверяет, что поисковая система исключает стоп-слова при добавлении документов
void TestExcludeStopWordsFromAddedDocumentContent() {
    const int doc_id = 42;
    const string content = "cat in the city"s;
    const vector<int> ratings = {1, 2, 3};
    {
        SearchServer server;
        server.AddDocument(doc_id, content, DocumentStatus::ACTUAL, ratings);
        const auto found_docs = server.FindTopDocuments("in"s);
        ASSERT_EQUAL(found_docs.size(), 1);
        const Document& doc0 = found_docs[0];
        ASSERT_EQUAL(doc0.id, doc_id);
    }

    {
        SearchServer server;
        server.SetStopWords("in the"s);
        server.AddDocument(doc_id, content, DocumentStatus::ACTUAL, ratings);
        ASSERT_HINT(server.FindTopDocuments("in"s).empty(), "Stop words must be excluded from documents"s);
    }
}

void TestSimpleSearch() {
    const vector<int> ratings = {1, 2, 3};
    SearchServer server;
    server.AddDocument(0, "cat in the big city"s, DocumentStatus::ACTUAL, ratings);
    server.AddDocument(1, "big developer in the big city", DocumentStatus::ACTUAL, ratings);
    server.AddDocument(2, "dog city", DocumentStatus::ACTUAL, ratings);
    server.AddDocument(3, "empty document", DocumentStatus::ACTUAL, ratings);
    server.AddDocument(4, "no text!!!!!!!!", DocumentStatus::ACTUAL, ratings);
    const auto found_docs = server.FindTopDocuments("city"s);
    ASSERT_HINT(found_docs.size() != 0, "Search returned zero documents"s);
    ASSERT_HINT(found_docs.size() >= 3, "Search does not return all the necessary documents"s);
    ASSERT_HINT(found_docs.size() == 3, "Search returns more documents than expected"s);
}

void TestMinusWordsExcludeDocFromSERP() {
    const vector<int> ratings = {1, 2, 3};
    int doc_id = 1;
    {
        SearchServer server;
        server.AddDocument(doc_id, "cat in the city"s, DocumentStatus::ACTUAL, ratings);
        const auto found_docs = server.FindTopDocuments("cat"s);
        ASSERT_EQUAL(found_docs.size(), 1);
        const Document& doc0 = found_docs[0];
        ASSERT_EQUAL(doc0.id, doc_id);
    }
    
    {
        SearchServer server;
        server.AddDocument(doc_id, "cat in the city"s, DocumentStatus::ACTUAL, ratings);
        server.AddDocument(2, "dog in the city", DocumentStatus::ACTUAL, ratings);
        const auto found_docs = server.FindTopDocuments("in the city -dog"s);
        ASSERT_EQUAL(found_docs.size(), 1);
        const Document& doc0 = found_docs[0];
        ASSERT_EQUAL_HINT(doc0.id, doc_id, "Document containing minus-words should be excluded from the search resultss"s);
    }
}

void TestRatingCalculation() {
    SearchServer server;
    server.AddDocument(0, "cat in the big city"s, DocumentStatus::ACTUAL, {1, 2, 3});
    server.AddDocument(1, "developer in the big chair", DocumentStatus::ACTUAL, {1, -2, 10});
    server.AddDocument(2, "dog city", DocumentStatus::ACTUAL, {-1, 2, -10});
    const auto found_docs = server.FindTopDocuments("in the big city"s);
    const Document& doc0 = found_docs[0];
    const Document& doc1 = found_docs[1];
    const Document& doc2 = found_docs[2];
    ASSERT_HINT(doc0.rating == 2, "Incorrect rating calculation"s);
    ASSERT_HINT(doc1.rating == 3, "Incorrect rating calculation"s);
    ASSERT_HINT(doc2.rating == -3, "Incorrect rating calculation in case of negative ratings"s);
}

void TestSearchForStatus() {
    const vector<int> ratings = {1, 2, 3};
    SearchServer server;
    server.AddDocument(0, "city and no text!!!!!!!!", DocumentStatus::REMOVED, ratings);
    server.AddDocument(1, "cat in the big city"s, DocumentStatus::ACTUAL, ratings);
    server.AddDocument(2, "big developer in the big city", DocumentStatus::ACTUAL, ratings);
    server.AddDocument(3, "city dog city", DocumentStatus::BANNED, ratings);
    server.AddDocument(4, "city and the empty document", DocumentStatus::IRRELEVANT, ratings);
    const auto found_docs = server.FindTopDocuments("city"s, DocumentStatus::ACTUAL);
    const Document& doc0 = found_docs[0];
    const Document& doc1 = found_docs[1];
    ASSERT_HINT(found_docs.size() == 2, "Search returns the wrong number of documents"s);
    ASSERT_HINT((doc0.id == 1 || doc0.id == 2), "Search returns documents with incorrect statuses"s);
    ASSERT_HINT((doc1.id == 1 || doc1.id == 2), "Search returns documents with incorrect statuses"s);
}

void TestUsersLambda() {
    const vector<int> ratings = {1, 2, 3};
    string text = "test text with no dark jokes";
    DocumentStatus status = DocumentStatus::ACTUAL;
    SearchServer server;
    server.AddDocument(0, text, status, ratings);
    server.AddDocument(1, text, status, ratings);
    server.AddDocument(2, text, status, ratings);
    server.AddDocument(3, text, status, ratings);
    server.AddDocument(4, text, status, ratings);
    const auto found_docs = server.FindTopDocuments("dark jokes"s, [](int document_id, DocumentStatus status[[maybe_unused]], int rating[[maybe_unused]]) { return document_id % 2 == 0; });
    ASSERT_HINT(found_docs.size() == 3, "Search returns the wrong number of documents"s);
    const Document& doc0 = found_docs[0];
    const Document& doc1 = found_docs[1];
    const Document& doc2 = found_docs[2];
    ASSERT_HINT(doc0.id % 2 == 0, "The condition from the custom lambda function is not met"s);
    ASSERT_HINT(doc1.id % 2 == 0, "The condition from the custom lambda function is not met"s);
    ASSERT_HINT(doc2.id % 2 == 0, "The condition from the custom lambda function is not met"s);
}

void TestRelevanceCalculationAndRelevanceSort() {
    const vector<int> ratings = {1, 2, 3};
    const double e = 1e-5; //погрешность
    double standard = 0.0;
    DocumentStatus status = DocumentStatus::ACTUAL;
    {
        SearchServer server;
        server.AddDocument(0, "test text with no dark jokes", status, ratings);
        const auto found_docs = server.FindTopDocuments("text"s);
        const Document& doc0 = found_docs[0];
        ASSERT(fabs(doc0.relevance - standard) < e);
        ASSERT_HINT(fabs(doc0.relevance - standard) < e, /* idf = 0, tf = 0.25 */ "Incorrect calculation of relevance in the presence of one document in the search engine"s);
        
    }
    
    {
        SearchServer server;
        server.AddDocument(0, "test text with no dark jokes text", status, ratings);
        server.AddDocument(1, "empty jar", status, ratings);
        server.AddDocument(3, "jar with nutella", status, ratings);
        const auto found_docs = server.FindTopDocuments("text with"s);
        const Document& doc0 = found_docs[0];
        const Document& doc1 = found_docs[1];
        standard = 0.371813;
        ASSERT_HINT(fabs(doc0.relevance - standard) < e, "Incorrect calculation of relevance"s);
        standard = 0.135155;
        ASSERT_HINT(fabs(doc1.relevance - standard) < e, "Incorrect calculation of relevance"s);

    }
}

void TestMatching() {
    const vector<int> ratings = {1, 2, 3};
    DocumentStatus status = DocumentStatus::ACTUAL;
    {
        SearchServer server;
        server.AddDocument(0, "test text with no dark jokes", status, ratings);
        vector<string> words;
        tie(words, status) = server.MatchDocument("test dark jokes"s, 0);
        ASSERT_HINT(words.size() == 3, "Matching returns the wrong number of words"s);
        ASSERT_HINT(words[2] == "test" || words[1] == "test" || words[0] == "test", "Incorrect word returned"s);
        ASSERT_HINT(words[1] == "dark" || words[2] == "dark" || words[0] == "dark", "Incorrect word returned"s);
        ASSERT_HINT(words[0] == "jokes" || words[1] == "jokes" || words[2] == "jokes", "Incorrect word returned"s);
    }
    
    {
        SearchServer server;
        server.AddDocument(0, "test text with no dark jokes", status, ratings);
        vector<string> words;
        tie(words, status) = server.MatchDocument("test -dark jokes"s, 0);
        ASSERT_HINT(words.size() == 0, "Matching should return an empty list of words if there is a negative word in the document"s);
    }
}

void TestSearchServer() {
    RUN_TEST(TestSimpleSearch);
    RUN_TEST(TestExcludeStopWordsFromAddedDocumentContent);
    RUN_TEST(TestMinusWordsExcludeDocFromSERP);
    RUN_TEST(TestRatingCalculation);
    RUN_TEST(TestSearchForStatus);
    RUN_TEST(TestUsersLambda);
    RUN_TEST(TestRelevanceCalculationAndRelevanceSort);
    RUN_TEST(TestMatching);
}

// --------- Окончание модульных тестов поисковой системы -----------

int main() {
    TestSearchServer();
    // Если вы видите эту строку, значит все тесты прошли успешно
    cout << "Search server testing finished"s << endl;
}
