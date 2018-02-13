from base.base_model import BaseModel


class TemplateModel(BaseModel):
    def __init__(self, config, data_loader):
        super(TemplateModel, self).__init__(config, data_loader)

    def _build_train_model(self):
        """
        define the train model.
        need to set:
        self.train_op, self.loss_op
        and add your summaries
        :return:
        """
        pass

    def _build_evaluate_model(self):
        """
        define the train model.
        need to set:
        self.train_op, self.loss_op
        and add your summaries
        :return:
        """
        pass
