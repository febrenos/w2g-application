import React from "react";
import FormSignup from "../../../components/FormSignup";
import Planet from "../../../components/Planet";
import TitleApresentation from "../../../components/TitleApresentation";
import * as Styled from '../style'

export default function(){
    return(
        <Styled.Content>
            <Styled.Apresentation>
                <div>
                    <Planet width={200}/>
                    <TitleApresentation/>
                </div>
            </Styled.Apresentation>
            <FormSignup/>
        </Styled.Content>
    )
}