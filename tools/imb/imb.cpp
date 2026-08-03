//===--- imb.cpp - Mulberry Interactive Shell ----------------------------===//

#include "isocline.h"
#include "mulberry/Driver/Compilation.h"
#include "mulberry/Driver/JitSession.h"
#include "mlir/IR/BuiltinOps.h"
#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

extern "C" void mulberry_runtime_init();
extern "C" void mulberry_runtime_enable_repl_print();

namespace {

const char *completionCandidates[] = {
    "const",       "var",          "mut",          "fn",
    "struct",      "data",         "trait",       "impl",
    "import",      "extern",       "comptime",    "match",
    "if",          "else",         "while",       "for",
    "break",       "continue",     "return",      "yield",
    "true",        "false",        "help",        "quit",
    "exit",        ":h",           ":help",       ":pwd",
    ":cd",          ":t",           ":type",       ":q",
    ":quit",        ":exit",        ":l",           ":load",
    "Bool",         "Char",        "UInt8",
    "UInt64",      "Float32",      "Integer",     "String",
    "Array",       "List",         "Tensor",      "Ptr",
    "Result",      "Ok",           "Err",         "Some",
    "None",        "File",         "Rng",         "Show",
    "core",        "io",           "list",        "random",
    "result",      "string",       "tensor",      "types",
    "bigint",      "json",         "safetensors", "print",
    "println",     "writeText",    "open",        "close",
    "read",        "readExact",    "write",       "withCapacity",
    "push",        "pop",          "map",         "filter",
    "reduce",      "forEach",      "toString",    "formatValue",
    "from",        "zeros",        "zerosDynamic", "reshape",
    "numel",       "size",         "identity",    "boolToUInt64",
    "toUInt8",     "toFloat32",    "sizeof",      "typeInfo",
    nullptr};

struct CompletionState {
  std::vector<std::string> semanticNames;
  std::vector<std::string> memberNames;
  std::vector<std::string> historyEntries;
  mulberry::Compilation *compilation = nullptr;
};

struct MemberCompletionInput {
  std::string receiver;
  std::string prefix;
};

auto isIdentifierCharacter(char character) -> bool {
  return (character >= 'a' && character <= 'z') ||
         (character >= 'A' && character <= 'Z') ||
         (character >= '0' && character <= '9') || character == '_';
}

auto isMemberPath(std::string_view path) -> bool {
  if (path.empty())
    return false;
  if (!((path.front() >= 'a' && path.front() <= 'z') ||
        (path.front() >= 'A' && path.front() <= 'Z') ||
        path.front() == '_'))
    return false;

  bool expectingIdentifier = false;
  for (size_t index = 1; index < path.size(); ++index) {
    if (path[index] == '.') {
      if (expectingIdentifier)
        return false;
      expectingIdentifier = true;
      continue;
    }
    if (!isIdentifierCharacter(path[index]))
      return false;
    expectingIdentifier = false;
  }
  return !expectingIdentifier;
}

auto memberCompletionInput(std::string_view input)
    -> std::optional<MemberCompletionInput> {
  auto dot = input.rfind('.');
  if (dot == std::string_view::npos)
    return std::nullopt;

  auto memberPrefix = input.substr(dot + 1);
  for (char character : memberPrefix)
    if (!isIdentifierCharacter(character))
      return std::nullopt;

  size_t receiverStart = dot;
  while (receiverStart != 0) {
    auto character = input[receiverStart - 1];
    if (!isIdentifierCharacter(character) && character != '.')
      break;
    --receiverStart;
  }
  auto receiver = input.substr(receiverStart, dot - receiverStart);
  if (!isMemberPath(receiver))
    return std::nullopt;
  return MemberCompletionInput{std::string(receiver),
                               std::string(memberPrefix)};
}

auto addCompletionCandidates(ic_completion_env_t *completion,
                             std::string_view prefix,
                             const std::vector<std::string> &candidates)
    -> void {
  std::set<std::string_view> emitted;
  for (const auto &candidate : candidates) {
    if (candidate.compare(0, prefix.size(), prefix) != 0)
      continue;
    if (!emitted.insert(candidate).second)
      continue;
    if (!ic_add_completion(completion, candidate.c_str()))
      return;
  }
}

auto hexDigit(char character) -> int {
  if (character >= '0' && character <= '9')
    return character - '0';
  if (character >= 'a' && character <= 'f')
    return character - 'a' + 10;
  if (character >= 'A' && character <= 'F')
    return character - 'A' + 10;
  return -1;
}

auto decodeHistoryEntry(std::string_view encoded)
    -> std::optional<std::string> {
  std::string decoded;
  for (size_t index = 0; index < encoded.size(); ++index) {
    auto character = encoded[index];
    if (character != '\\') {
      decoded += character;
      continue;
    }

    if (++index >= encoded.size())
      return std::nullopt;
    switch (encoded[index]) {
    case 'n':
      decoded += '\n';
      break;
    case 'r':
      // Isocline intentionally drops carriage returns when loading history.
      break;
    case 't':
      decoded += '\t';
      break;
    case '\\':
      decoded += '\\';
      break;
    case 'x': {
      if (index + 2 >= encoded.size())
        return std::nullopt;
      auto high = hexDigit(encoded[index + 1]);
      auto low = hexDigit(encoded[index + 2]);
      if (high < 0 || low < 0)
        return std::nullopt;
      decoded += static_cast<char>((high << 4) | low);
      index += 2;
      break;
    }
    default:
      return std::nullopt;
    }
  }
  return decoded;
}

auto loadHistoryEntries(const std::string &path) -> std::vector<std::string> {
  std::vector<std::string> entries;
  if (path.empty())
    return entries;

  std::ifstream file(path);
  std::string encoded;
  while (std::getline(file, encoded)) {
    auto decoded = decodeHistoryEntry(encoded);
    if (!decoded)
      break;
    if (decoded->empty() || decoded->front() == '#')
      continue;
    entries.push_back(std::move(*decoded));
  }

  // Isocline writes history from oldest to newest; completion should prefer
  // the newest matching command, just like shell autosuggestion does.
  std::reverse(entries.begin(), entries.end());
  return entries;
}

auto rememberHistoryEntry(CompletionState &state, std::string_view entry)
    -> void {
  if (entry.size() <= 1)
    return;

  auto existing = std::find(state.historyEntries.begin(),
                            state.historyEntries.end(), entry);
  if (existing != state.historyEntries.end())
    state.historyEntries.erase(existing);
  state.historyEntries.insert(state.historyEntries.begin(),
                              std::string(entry));
  if (state.historyEntries.size() > 200)
    state.historyEntries.pop_back();
}

auto addHistoryCompletion(ic_completion_env_t *completion, const char *prefix)
    -> bool {
  auto *state = static_cast<const CompletionState *>(
      ic_completion_arg(completion));
  if (state == nullptr || prefix == nullptr || *prefix == '\0')
    return false;

  std::string_view input(prefix);
  for (const auto &entry : state->historyEntries) {
    if (entry.size() <= input.size() ||
        entry.compare(0, input.size(), input) != 0)
      continue;

    // `ic_add_completion_prim` replaces the whole input, so Isocline can
    // render `entry.substr(input.size())` as its normal gray hint and accept
    // it with the existing right-arrow behavior.
    return ic_add_completion_prim(completion, entry.c_str(), entry.c_str(),
                                  nullptr, static_cast<long>(input.size()), 0);
  }
  return false;
}

auto addMulberryCompletions(ic_completion_env_t *completion,
                            const char *prefix) -> void {
  auto *state = static_cast<const CompletionState *>(
      ic_completion_arg(completion));
  std::string_view prefixView = prefix == nullptr ? std::string_view{} : prefix;
  std::set<std::string_view> emitted;

  auto add = [&](std::string_view candidate) -> bool {
    if (candidate.compare(0, prefixView.size(), prefixView) != 0)
      return true;
    if (!emitted.insert(candidate).second)
      return true;
    return ic_add_completion(completion, candidate.data());
  };

  for (const char *candidate : completionCandidates) {
    if (candidate == nullptr)
      break;
    if (!add(candidate))
      return;
  }

  if (state == nullptr)
    return;
  for (const auto &candidate : state->semanticNames)
    if (!add(candidate))
      return;
}

auto addMulberryMemberCompletions(ic_completion_env_t *completion,
                                  const char *prefix) -> void {
  auto *state = static_cast<const CompletionState *>(
      ic_completion_arg(completion));
  if (state == nullptr)
    return;
  addCompletionCandidates(completion,
                           prefix == nullptr ? std::string_view{} : prefix,
                           state->memberNames);
}

auto completeMulberryWord(ic_completion_env_t *completion,
                          const char *prefix) -> void {
  auto *state = static_cast<CompletionState *>(
      ic_completion_arg(completion));
  if (addHistoryCompletion(completion, prefix))
    return;

  // The default Isocline completer receives the untransformed input up to the
  // cursor. Keep this source-aware path independent of optional advanced APIs.
  if (state != nullptr && state->compilation != nullptr && prefix != nullptr) {
    auto member = memberCompletionInput(prefix);
    if (member) {
      state->memberNames =
          state->compilation->replCompletionMembers(member->receiver);
      ic_complete_word(completion, member->prefix.c_str(),
                       addMulberryMemberCompletions, nullptr);
      return;
    }
  }
  ic_complete_word(completion, prefix, addMulberryCompletions, nullptr);
}

auto historyPath() -> std::string {
  const char *home = std::getenv("HOME");
  if (home == nullptr || *home == '\0')
    return {};

  std::filesystem::path directory(home);
  directory /= ".mulberry";

  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error)
    return {};

  return (directory / "mulberry-history").string();
}

auto needsMoreInput(std::string_view source) -> bool {
  unsigned parentheses = 0;
  unsigned brackets = 0;
  unsigned braces = 0;
  char stringDelimiter = 0;
  bool escaped = false;
  bool lineComment = false;

  for (size_t index = 0; index < source.size(); ++index) {
    auto character = source[index];
    if (lineComment) {
      if (character == '\n')
        lineComment = false;
      continue;
    }
    if (stringDelimiter != 0) {
      if (escaped) {
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == stringDelimiter) {
        stringDelimiter = 0;
      }
      continue;
    }
    if (character == '/' && index + 1 < source.size() &&
        source[index + 1] == '/') {
      lineComment = true;
      ++index;
      continue;
    }
    if (character == '"' || character == '\'') {
      stringDelimiter = character;
      continue;
    }
    switch (character) {
    case '(':
      ++parentheses;
      break;
    case ')':
      if (parentheses != 0)
        --parentheses;
      break;
    case '[':
      ++brackets;
      break;
    case ']':
      if (brackets != 0)
        --brackets;
      break;
    case '{':
      ++braces;
      break;
    case '}':
      if (braces != 0)
        --braces;
      break;
    default:
      break;
    }
  }

  if (parentheses != 0 || brackets != 0 || braces != 0 ||
      stringDelimiter != 0)
    return true;

  auto end = source.find_last_not_of(" \t\r\n");
  if (end == std::string_view::npos)
    return false;
  auto last = source[end];
  return last == '+' || last == '-' || last == '*' || last == '/' ||
         last == '%' || last == '=' || last == ',' || last == '.' ||
         last == ':';
}

auto readSubmission() -> std::optional<std::string> {
  char *input = ic_readline("imb ");
  if (input == nullptr)
    return std::nullopt;

  std::string source(input);
  ic_free(input);
  while (needsMoreInput(source)) {
    char *continuation = ic_readline("... ");
    if (continuation == nullptr)
      return std::nullopt;
    // Isocline represents Ctrl-C as an empty line. Treat an empty continuation
    // as cancellation so a partial submission is never compiled accidentally.
    if (*continuation == '\0') {
      ic_free(continuation);
      return std::string{};
    }
    source += '\n';
    source += continuation;
    ic_free(continuation);
  }
  return source;
}

struct InternalCommand {
  std::string_view name;
  std::string_view argument;
};

enum class InternalCommandResult { NotCommand, Continue, Exit, Load };

auto trimWhitespace(std::string_view value) -> std::string_view {
  auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string_view::npos)
    return {};
  auto end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

auto parseInternalCommand(std::string_view line)
    -> std::optional<InternalCommand> {
  if (line.empty() || line.front() != ':')
    return std::nullopt;

  auto separator = line.find_first_of(" \t\r\n");
  if (separator == std::string_view::npos)
    return InternalCommand{line, {}};

  return InternalCommand{line.substr(0, separator),
                         trimWhitespace(line.substr(separator + 1))};
}

auto printCommandUsage(std::string_view command) -> bool {
  if (!command.empty() && command.front() == ':')
    command.remove_prefix(1);

  if (command == "h" || command == "help") {
    std::cout << ":help [command]\n";
    return true;
  }
  if (command == "pwd") {
    std::cout << ":pwd\n";
    return true;
  }
  if (command == "cd") {
    std::cout << ":cd <path>\n";
    return true;
  }
  if (command == "t" || command == "type") {
    std::cout << ":type <expression>\n";
    return true;
  }
  if (command == "l" || command == "load") {
    std::cout << ":load <path>\n";
    return true;
  }
  if (command == "q" || command == "quit" || command == "exit") {
    std::cout << ":quit\n";
    return true;
  }
  return false;
}

auto printHelp(std::string_view argument) -> void {
  if (argument.empty()) {
    std::cout << ":h, :help [command]       show commands or command usage\n"
                 ":pwd                      print the current working directory\n"
                 ":cd <path>                change the current working directory\n"
                 ":t, :type <expression>    print an expression's type\n"
                 ":l, :load <path>          load a file into the current session\n"
                 ":q, :quit, :exit          exit imb\n";
    return;
  }

  if (argument.find_first_of(" \t\r\n") != std::string_view::npos) {
    std::cerr << "imb: usage: :help [command]\n";
    return;
  }

  if (!printCommandUsage(argument))
    std::cerr << "imb: unknown command '" << argument << "'\n";
}

auto readLoadFile(std::string_view argument, std::string &source,
                  std::string &sourceName) -> bool {
  std::filesystem::path path(argument);
  std::error_code error;
  path = std::filesystem::absolute(path, error);
  if (error) {
    std::cerr << "imb: load: " << error.message() << ": '" << argument
              << "'\n";
    return false;
  }

  std::ifstream file(path, std::ios::binary);
  if (!file) {
    std::cerr << "imb: load: unable to read '" << path.string() << "'\n";
    return false;
  }

  source.assign(std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>());
  if (file.bad()) {
    std::cerr << "imb: load: failed while reading '" << path.string()
              << "'\n";
    return false;
  }

  sourceName = path.lexically_normal().string();
  return true;
}

auto handleInternalCommand(std::string_view line,
                           mulberry::Compilation &compilation,
                           std::string &loadedSource,
                           std::string &loadedSourceName)
    -> InternalCommandResult {
  auto command = parseInternalCommand(line);
  if (!command)
    return InternalCommandResult::NotCommand;

  if (command->name == ":q" || command->name == ":quit" ||
      command->name == ":exit") {
    if (!command->argument.empty())
      std::cerr << "imb: usage: :quit\n";
    else
      return InternalCommandResult::Exit;
    return InternalCommandResult::Continue;
  }

  if (command->name == ":h" || command->name == ":help") {
    printHelp(command->argument);
    return InternalCommandResult::Continue;
  }

  if (command->name == ":pwd") {
    if (!command->argument.empty()) {
      std::cerr << "imb: usage: :pwd\n";
      return InternalCommandResult::Continue;
    }
    std::error_code error;
    auto path = std::filesystem::current_path(error);
    if (error)
      std::cerr << "imb: pwd: " << error.message() << "\n";
    else
      std::cout << path.string() << '\n';
    return InternalCommandResult::Continue;
  }

  if (command->name == ":cd") {
    if (command->argument.empty()) {
      std::cerr << "imb: usage: :cd <path>\n";
      return InternalCommandResult::Continue;
    }
    std::string path(command->argument);
    std::error_code error;
    std::filesystem::current_path(path, error);
    if (error)
      std::cerr << "imb: cd: " << error.message() << ": '" << path
                << "'\n";
    return InternalCommandResult::Continue;
  }

  if (command->name == ":t" || command->name == ":type") {
    if (command->argument.empty()) {
      std::cerr << "imb: usage: :type <expression>\n";
      return InternalCommandResult::Continue;
    }
    std::string typeName;
    llvm::StringRef expression(command->argument.data(),
                               command->argument.size());
    if (llvm::succeeded(
            compilation.replTypeOf(expression, "<imb:type>", typeName)))
      std::cout << typeName << '\n';
    return InternalCommandResult::Continue;
  }

  if (command->name == ":l" || command->name == ":load") {
    if (command->argument.empty()) {
      std::cerr << "imb: usage: :load <path>\n";
      return InternalCommandResult::Continue;
    }
    if (!readLoadFile(command->argument, loadedSource, loadedSourceName))
      return InternalCommandResult::Continue;
    return InternalCommandResult::Load;
  }

  std::cerr << "imb: unknown command '" << command->name << "'\n";
  return InternalCommandResult::Continue;
}

} // namespace

auto main() -> int {
  std::setlocale(LC_ALL, "");
  ic_init(false);

  const std::string history = historyPath();
  ic_set_history(history.empty() ? nullptr : history.c_str(), -1);
  ic_set_prompt_marker("> ", ". ");
  ic_enable_completion_preview(true);
  ic_enable_multiline_indent(true);
  ic_enable_hint(true);
  ic_set_hint_delay(250);
  ic_enable_history_duplicates(false);

  CompletionState completionState;
  completionState.historyEntries = loadHistoryEntries(history);

  auto compilation = mulberry::Compilation::make("-", false);
  if (compilation == nullptr)
    return EXIT_FAILURE;
  completionState.compilation = compilation.get();
  ic_set_default_completer(completeMulberryWord, &completionState);

  auto jit = mulberry::JitSession::create(false);
  if (!jit) {
    std::cerr << "imb: failed to create JIT session: "
              << llvm::toString(jit.takeError()) << '\n';
    return EXIT_FAILURE;
  }

  mulberry_runtime_init();
  mulberry_runtime_enable_repl_print();
  uint64_t submission = 0;

  while (true) {
    auto input = readSubmission();
    if (!input) {
      std::cout << '\n';
      break;
    }

    auto line = std::move(*input);
    if (line.empty()) {
      ic_history_remove_last();
      continue;
    }

    rememberHistoryEntry(completionState, line);

    std::string sourceName = "<imb>";
    std::string loadedSource;
    auto commandResult = handleInternalCommand(
        line, *compilation, loadedSource, sourceName);
    if (commandResult == InternalCommandResult::Exit)
      break;
    if (commandResult == InternalCommandResult::Continue)
      continue;
    if (commandResult == InternalCommandResult::Load)
      line = std::move(loadedSource);

    auto functionName = "__imb_" + std::to_string(submission);
    auto moduleName = "imb_submission_" + std::to_string(submission);
    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (llvm::failed(compilation->compileReplSource(
            line, sourceName, functionName, module,
            mulberry::Compilation::Lowering::LLVM))) {
      ++submission;
      continue;
    }

    if (auto error = (*jit)->addModule(*module, moduleName)) {
      compilation->rollbackReplSubmission();
      std::cerr << "imb: failed to add submission: "
                << llvm::toString(std::move(error)) << '\n';
      ++submission;
      continue;
    }
    if (auto error = (*jit)->initialize(moduleName)) {
      compilation->rollbackReplSubmission();
      std::cerr << "imb: failed to initialize submission: "
                << llvm::toString(std::move(error)) << '\n';
      ++submission;
      continue;
    }

    uint64_t result = 0;
    if (auto error = (*jit)->invoke(moduleName, functionName, result)) {
      compilation->rollbackReplSubmission();
      std::cerr << "imb: evaluation failed: "
                << llvm::toString(std::move(error)) << '\n';
      ++submission;
      continue;
    }

    compilation->commitReplSubmission();
    // Runtime print uses C stdio; flush it before the next REPL prompt so a
    // print without a trailing newline is visible immediately.
    std::fflush(stdout);
    completionState.semanticNames = compilation->replCompletionNames();
    ++submission;
  }

  return 0;
}
