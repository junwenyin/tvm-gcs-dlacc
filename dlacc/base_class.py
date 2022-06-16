import datetime


class BaseClass:
    def dump_object(self):
        attrs = vars(self)
        content = ""
        if len(attrs):
            content = ", ".join("%s: %s" % item for item in attrs.items())
        return content

    def raise_default_error(self, content=""):
        raise RuntimeError(content + "\n" + self.dump_object())

    def _print(self, content):
        print(
            "[%s][%s]"
            % (
                datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S"),
                self.__class__.__name__,
            ),
            end=" ",
        )
        print(content)
