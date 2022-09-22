class GPT2LMHeadModel(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(self, config):
        super().__init__(config)
        # 初始化GPT2Model(config)类.
        self.transformer = GPT2Model(config)

        # self.lm_head为将GPT2Model(config)计算输出的hidden_states张量的最后一个维度由768维(config.n_embd)投影为
        # 词典大小维度(config.vocab_size)的输出层, 此时hidden_states张量的形状将会由(batch_size, 1, n_embed)投影变为
        # lm_logits张量的(batch_size, 1, vocab_size).
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 重新初始化权重矩阵.
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="gpt2",
        output_type=CausalLMOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 此时返回的transformer_outputs中为：
        # <1> 第一个值为GPT2模型中经过12层Block模块计算后得到的最终hidden_states张量,
        #     形状为(batch_size, 1, n_state), all_head_size=n_state=nx=n_embd=768.
        # <2> 第二个值为GPT2模型中12层Block模块计算后得到的存储12个present张量的presents元组, 每一个present张量存储着
        #     past_key张量与这次迭代的key张量合并后的新key张量, 以及past_value张量与这次迭代的value张量合并后的新value张量,
        #     一个present张量形状为(2, batch_size, num_head, sql_len+1, head_features).
        # <3> 若output_hidden_states为True, 则第三个值为GPT2模型中12层Block模块计算后得到的存储12个隐藏状态张量hidden_states
        #     的all_hidden_states元组.
        # <4> 若output_attentions为True, 则第四个值为GPT2模型中12层Block模块计算后得到的存储12个注意力分数张量w
        #     的all_self_attentions元组.
        # <5> 若此时进行了Cross Attention计算, 则第五个值为GPT2模型中12层Block模块计算后得到的存储12个交叉注意力分数张量
        #     cross_attention的all_cross_attentions元组,
        #     其中每个交叉注意力分数张量cross_attention形状为(batch_size, num_head, 1, enc_seq_len).
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # self.lm_head()输出层将GPT2Model(config)计算输出的hidden_states张量的最后一个维度由768维(config.n_embd)
        # 投影为词典大小维度(config.vocab_size)的输出层, 此时hidden_states张量的形状将会由(batch_size, 1, n_embed)投影变为
        # lm_logits张量的(batch_size, 1, vocab_size).
        lm_logits = self.lm_head(hidden_states)

        loss = None
        # 若此时labels也输入进了GPT2LMHeadModel()类中, 则此时会使用自回归的方式计算交叉熵损失,
        # 即此时的shift_logits为将GPT2Model(config)计算输出的hidden_states张量的最后一个维度由768维(config.n_embd)投影为
        # 词典大小维度(config.vocab_size)所得到的lm_logits张量的切片lm_logits[..., :-1, :].contiguous(),即取(1, n-1)的lm_logits值；
        # 此时的shift_labels为将输入的labels张量的切片labels[..., 1:].contiguous(), 即取(2, n)的label值；
        # 因此利用(1, n-1)的lm_logits值与(2, n)的label值即可计算此时自回归预训练的交叉熵损失值.
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # <1> 若loss不为None, 则代表此时输入了labels张量, 进行了自回归的交叉熵损失计算, 则此时第一个值为
        #     自回归交叉熵损失loss.
        # <2> 第二个值将GPT2Model(config)计算输出的hidden_states张量的最后一个维度由768维(config.n_embd)投影为
        #     词典大小维度(config.vocab_size)的lm_logits张量, 其形状为(batch_size, 1, vocab_size).
        # <3> 第三个值为GPT2模型中12层Block模块计算后得到的存储12个present张量的presents元组, 每一个present张量存储着
        #     past_key张量与这次迭代的key张量合并后的新key张量, 以及past_value张量与这次迭代的value张量合并后的新value张量,
        #     一个present张量形状为(2, batch_size, num_head, sql_len+1, head_features).
        # <4> 若output_hidden_states为True, 则第四个值为GPT2模型中12层Block模块计算后得到的存储12个隐藏状态张量hidden_states
        #     的all_hidden_states元组.
        # <5> 若output_attentions为True, 则第五个值为GPT2模型中12层Block模块计算后得到的存储12个注意力分数张量w
        #     的all_self_attentions元组.
        # <6> 若此时进行了Cross Attention计算, 则第六个值为GPT2模型中12层Block模块计算后得到的存储12个交叉注意力分数张量
        #     cross_attention的all_cross_attentions元组,
        #     其中每个交叉注意力分数张量cross_attention形状为(batch_size, num_head, 1, enc_seq_len).
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPastAndCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
