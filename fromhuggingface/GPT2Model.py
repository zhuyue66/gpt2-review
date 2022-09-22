class GPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # position embedding
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="gpt2",
        output_type=BaseModelOutputWithPastAndCrossAttentions,
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
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # input_ids与inputs_embeds只能输入一个，有input_ids变只需将input_ids输入嵌入层即可变为类似inputs_embeds的张量,
        # 有inputs_embeds变不需要input_ids
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")

        # 下方是确保输入的input_ids、token_type_ids、position_ids等张量的形状为正确的样式:
        # <1> 若为模型第一次迭代, 则此时input_ids、token_type_ids、position_ids等张量的正确形状为 (batch_size, seq_len),
        # <2> 若为模型第二次及之后的迭代, 则此时input_ids、token_type_ids、position_ids等张量的正确形状为 (batch_size, 1).
        # 最后, 将输入的input_ids、token_type_ids、position_ids等张量的形状保存到input_shape中.
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            # 若此时为GPT2模型第一次迭代, 则不存在上一次迭代返回的past_key_values列表(包含12个present的列表,
            # 也就是代码中的presents列表), 则此时past_key_values列表为一个包含12个None值的列表.
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            '''<1> GPT2Model第一次迭代时输入GPT2Model的forward()函数中的past_key_values参数为None, 此时past_length为0, 
              input_shape[-1] + past_length就等于第一次迭代时输入的文本编码(input_ids)的seq_len维度本身, 
              此时创建的position_ids张量形状为(batch_size, seq_len).
              <2> 若为GPT2Mode第二次及之后的迭代时, 此时past_length为上一次迭代时记录保存下来的past_key_values中
              张量的seq_len维度, 而input_shape[-1] + past_length则等于seq_len + 1, 因为在第二次及之后的迭代中,
              输入的文本编码(input_ids)的seq_len维度本身为1,即第二次及之后的迭代中每次只输入一个字的文本编码,
              此时创建的position_ids张量形状为(batch_size, 1).'''
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        # attention_mask张量为注意力遮罩张量, 其让填充特殊符[PAD]处的注意力分数极小,其embedding嵌入值
        # 基本不会在多头注意力聚合操作中被获取到.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length],
        # 若此时有从编码器encoder中传入的编码器隐藏状态encoder_hidden_states, 则获取编码器隐藏状态encoder_hidden_states
        # 的形状(encoder_batch_size, encoder_sequence_length), 同时定义编码器隐藏状态对应的attention_mask张量(即encoder_attention_mask).
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        # prune_heads()可结合 https://github.com/huggingface/transformers/issues/850 理解.
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        # 将input_ids、token_type_ids、position_ids等张量输入嵌入层self.wte()、 self.wpe()中之后获取其嵌入形式张量
        # inputs_embeds、position_embeds与token_type_embeds.
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        '''<1> GPT2Model第一次迭代时输入GPT2Model的forward()函数中的past_key_values参数为None, 此时past_length为0, 
              此时hidden_states张量形状为(batch_size, sel_len, n_embd)，config的GPT2Config()类中n_emb默认为768.
          <2> 若为GPT2Mode第二次及之后的迭代时, 此时past_length为上一次迭代时记录保存下来的past_key_values中
              张量的seq_len维度, 而input_shape[-1] + past_length则等于seq_len + 1, 因为在第二次及之后的迭代中,
              输入的文本编码(input_ids)的seq_len维度本身为1,即第二次及之后的迭代中每次只输入一个字的文本编码,
              此时hidden_states张量形状为(batch_size, 1, n_embd)，config的GPT2Config()类中n_emb默认为768.'''
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        # config对应的GPT2Config()类中的use_cache默认为True.
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            '''此处past_key_values元组中一共有12个元素(layer_past), 分别对应GPT2模型中的12层Transformer_Block,
            每一个layer_past都为模型上一次迭代中每个Transformer_Block保留下来的present张量, 而每个present张量保存着
            Transformer_Block中Attention模块将本次迭代的key张量与上一次迭代中的past_key张量(layer_past[0])合并、
            将本次迭代的value张量与上一次迭代中的past_value张量(layer_past[1])合并所得的新的key张量与value张量,
            之后保存着本次迭代中12层Transformer_Block每一层中返回的present张量的presents元组, 便会被作为下一次迭代中
            的past_key_values元组输入进下一次迭代的GPT2模型中。
            新的key张量与value张量详细解析如下：'''

            '''第一次迭代时query、key、value张量的seq_len维度处的维度数就为seq_len而不是1, 第二次之后seq_len维度的维度数皆为1.'''

            '''<1> 本次迭代中新的key张量
            此时需要通过layer_past[0].transpose(-2, -1)操作将past_key张量的形状变为(batch_size, num_head, head_features, sql_len),
            而此时key张量的形状为(batch_size, num_head, head_features, 1), 这样在下方就方便将past_key张量与key张量在最后
            一个维度(dim=-1)处进行合并, 这样就将当前token的key部分加入了past_key的seq_len部分, 以方便模型在后面预测新的token,
            此时新的key张量的形状为: (batch_size, num_head, head_features, sql_len+1), new_seq_len为sql_len+1。
             <2>  本次迭代中新的value张量
            而此时past_value(layer_past[1])不用变形, 其形状为(batch_size, num_head, sql_len, head_features), 
            而此时value张量的形状为(batch_size, num_head, 1, head_features), 这样在下方就方便将past_value张量与value张量
            在倒数第二个维度(dim=-2)处进行合并, 这样就将当前token的value部分加入了past_value的seq_len部分, 
            以方便模型在后面预测新的token,
            此时新的value张量的形状为: (batch_size, num_head, sql_len+1, head_features), new_seq_len为sql_len+1。'''

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # checkpointing only works with tuple returns, not with lists
                        return tuple(output for output in module(*inputs, use_cache, output_attentions))

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    layer_past,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                # 此时返回的outputs列表中的元素为：
                # <1> 第一个值为多头注意力聚合操作结果张量hidden_states输入前馈MLP层与残差连接之后得到的hidden_states张量,
                #     形状为(batch_size, 1, n_state), all_head_size=n_state=nx=n_embd=768.
                # <2> 第二个值为上方的present张量, 其存储着past_key张量与这次迭代的key张量合并后的新key张量, 以及
                #     past_value张量与这次迭代的value张量合并后的新value张量, 其形状为(2, batch_size, num_head, sql_len+1, head_features).
                # <3> 若output_attentions为True, 则第三个值为attn_outputs列表中的注意力分数张量w.
                # <4> 若此时进行了Cross Attention计算, 则第四个值为'交叉多头注意力计算结果列表cross_attn_outputs'中的
                #     交叉注意力分数张量cross_attention, 其形状为(batch_size, num_head, 1, enc_seq_len).
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3],)

        # 将PT2模型中12层Block模块计算后得到的最终hidden_states张量再输入进LayerNormalization层中进行计算.
        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state, 即将上方最后一层Block()循环结束之后得到的结果隐藏状态张量hidden_states
        # 也添加入元组all_hidden_states中.
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 此时返回的元素为：
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
        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
